import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

/** Reads the canonical Train/Validation/Test ARFF files, runs D12, and writes ARFF/CSV outputs. */
public final class D12Runner {
    private static final Map<String, Integer> EXPECTED_ROWS = Map.of(
            "Train", 44_794, "Validation", 11_198, "Test", 13_998);
    private static final String POSITIVE_LABEL = "readmitted";

    private D12Runner() {}

    /** Args: [input-dir] [arff-output-dir] [tables-output-dir] [positive-label]. */
    public static void main(String[] args) throws Exception {
        Path inputDir = args.length > 0 ? Path.of(args[0]) : Path.of("data/weka");
        Path arffDir = args.length > 1 ? Path.of(args[1]) : Path.of("data/weka/d12");
        Path tablesDir = args.length > 2 ? Path.of(args[2]) : Path.of("outputs/tables/data_mining");
        String positiveLabel = args.length > 3 ? args[3] : POSITIVE_LABEL;

        Map<String, Instances> raw = new LinkedHashMap<>();
        for (String split : List.of("Train", "Validation", "Test")) {
            String fileName = split.toLowerCase() + ".arff";
            Path file = inputDir.resolve(fileName);
            if (!Files.isRegularFile(file)) throw new IOException("Missing canonical input " + file.toAbsolutePath());
            Instances data = DataSource.read(file.toString());
            if (data == null) throw new IOException("Weka could not read " + file.toAbsolutePath());
            data.setClassIndex(data.numAttributes() - 1);
            validateCanonicalSchema(data, split);
            int expected = EXPECTED_ROWS.get(split);
            if (data.numInstances() != expected) {
                throw new IllegalArgumentException(split + " has " + data.numInstances() + " rows; canonical D12 expects " + expected);
            }
            raw.put(split, data);
        }
        Instances train = raw.get("Train");
        for (String split : List.of("Validation", "Test")) {
            if (!train.equalHeaders(raw.get(split))) {
                throw new IllegalArgumentException(split + " header is incompatible with Train: " + train.equalHeadersMsg(raw.get(split)));
            }
        }
        Attribute classAttribute = train.classAttribute();
        int positiveIndex = classAttribute.indexOfValue(positiveLabel);
        if (positiveIndex < 0) throw new IllegalArgumentException("Class attribute has no positive label '" + positiveLabel + "'");

        D12MixedDiscretizer discretizer = new D12MixedDiscretizer().fit(train);
        Map<String, Instances> transformed = new LinkedHashMap<>();
        Files.createDirectories(arffDir);
        for (Map.Entry<String, Instances> entry : raw.entrySet()) {
            Instances output = discretizer.transform(entry.getValue());
            if (output.numInstances() != entry.getValue().numInstances()) throw new IllegalStateException(entry.getKey() + " row count changed during transform");
            validateTransformPreservation(entry.getValue(), output, discretizer, entry.getKey());
            transformed.put(entry.getKey(), output);
            writeArff(output, arffDir.resolve(entry.getKey().toLowerCase() + ".arff"));
        }
        if (!transformed.get("Train").equalHeaders(transformed.get("Validation"))
                || !transformed.get("Train").equalHeaders(transformed.get("Test"))) {
            throw new IllegalStateException("D12 output headers are not compatible across splits");
        }
        verifyArffRoundTrips(transformed, arffDir);

        Files.createDirectories(tablesDir);
        writeCsv(tablesDir.resolve("d12_cut_points.csv"), cutPointRows(discretizer));
        writeCsv(tablesDir.resolve("d12_bin_summary.csv"), binSummaryRows(discretizer, transformed));
        writeCsv(tablesDir.resolve("d12_b1_rates.csv"), b1Rows(transformed, positiveLabel, discretizer));
        writeCsv(tablesDir.resolve("d12_split_audit.csv"), splitAuditRows(raw, positiveLabel));
        writeCsv(tablesDir.resolve("d12_b2_status.csv"), b2StatusRows(inputDir));

        System.out.println("D12 completed.");
        System.out.println("ARFF output: " + arffDir.toAbsolutePath());
        System.out.println("Tables: " + tablesDir.toAbsolutePath());
        for (String split : raw.keySet()) {
            System.out.printf("%s: %d rows, %d attributes (32 predictors + class)%n", split, raw.get(split).numInstances(), raw.get(split).numAttributes());
        }
        for (Map.Entry<String, double[]> entry : discretizer.cutPoints().entrySet()) {
            System.out.println(entry.getKey() + " (" + D12MixedDiscretizer.config().get(entry.getKey()).strategy() + "): " + formatCuts(entry.getValue()));
        }
        System.out.println("B1 rates: written for Train and Validation.");
        System.out.println("B2 rates: blocked; see d12_b2_status.csv for the verified input requirement.");
    }

    private static void validateCanonicalSchema(Instances data, String split) {
        if (data.numAttributes() != 33) {
            throw new IllegalArgumentException(split + " must contain 32 predictors and one final class column; found " + data.numAttributes() + " attributes");
        }
        if (!"class".equals(data.classAttribute().name())) throw new IllegalArgumentException(split + " final attribute must be named 'class'");
        if (!data.classAttribute().isNominal()) throw new IllegalArgumentException(split + " class must be nominal");
        for (String feature : D12MixedDiscretizer.config().keySet()) {
            Attribute attribute = data.attribute(feature);
            if (attribute == null) throw new IllegalArgumentException(split + " is missing D12 feature '" + feature + "'");
            if (!attribute.isNumeric()) throw new IllegalArgumentException(split + " D12 feature '" + feature + "' must be numeric");
        }
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attribute = data.attribute(i);
            if (!attribute.isNumeric() && !attribute.isNominal()) {
                throw new IllegalArgumentException(split + " attribute '" + attribute.name() + "' must be numeric or nominal");
            }
        }
    }

    private static void writeArff(Instances data, Path path) throws Exception {
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(path.toFile());
        saver.writeBatch();
    }

    private static void verifyArffRoundTrips(Map<String, Instances> expected, Path arffDir) throws Exception {
        Instances trainReloaded = null;
        for (String split : expected.keySet()) {
            Instances actual = DataSource.read(arffDir.resolve(split.toLowerCase() + ".arff").toString());
            actual.setClassIndex(actual.numAttributes() - 1);
            Instances original = expected.get(split);
            if (actual.numInstances() != original.numInstances() || !actual.equalHeaders(original)) {
                throw new IllegalStateException(split + " ARFF round-trip changed rows or header");
            }
            for (int r = 0; r < original.numInstances(); r++) {
                for (int a = 0; a < original.numAttributes(); a++) {
                    if (original.instance(r).isMissing(a) != actual.instance(r).isMissing(a)) {
                        throw new IllegalStateException(split + " ARFF round-trip changed missingness at row " + r + ", attribute " + a);
                    }
                    if (!original.instance(r).isMissing(a) && original.instance(r).value(a) != actual.instance(r).value(a)) {
                        throw new IllegalStateException(split + " ARFF round-trip changed value at row " + r + ", attribute " + a);
                    }
                }
            }
            if (trainReloaded == null) trainReloaded = actual;
            else if (!trainReloaded.equalHeaders(actual)) throw new IllegalStateException("Round-trip headers differ for " + split);
        }
    }

    private static void validateTransformPreservation(Instances input, Instances output,
            D12MixedDiscretizer discretizer, String split) {
        if (input.numAttributes() != output.numAttributes() || input.classIndex() != output.classIndex()) {
            throw new IllegalStateException(split + " transform changed attribute count or class index");
        }
        for (int r = 0; r < input.numInstances(); r++) {
            Instance before = input.instance(r), after = output.instance(r);
            if (before.weight() != after.weight()) throw new IllegalStateException(split + " transform changed instance weight at row " + r);
            for (int a = 0; a < input.numAttributes(); a++) {
                String name = input.attribute(a).name();
                if (!name.equals(output.attribute(a).name())) throw new IllegalStateException(split + " transform changed attribute order at row " + r);
                double[] cuts = discretizer.cutPoints().get(name);
                if (cuts == null) {
                    if (before.isMissing(a) != after.isMissing(a)) throw new IllegalStateException(split + " changed missingness for " + name + " at row " + r);
                    if (!before.isMissing(a) && before.value(a) != after.value(a)) throw new IllegalStateException(split + " changed value for " + name + " at row " + r);
                } else {
                    if (before.isMissing(a)) {
                        if (!after.isMissing(a)) throw new IllegalStateException(split + " changed missingness for " + name + " at row " + r);
                        continue;
                    }
                    double value = before.value(a);
                    if (value == Double.POSITIVE_INFINITY) {
                        if (!after.isMissing(a)) throw new IllegalStateException(split + " did not map +Infinity to missing for " + name + " at row " + r);
                    } else {
                        if (after.isMissing(a)) throw new IllegalStateException(split + " changed missingness for " + name + " at row " + r);
                        int expectedBin = value == Double.NEGATIVE_INFINITY ? 0 : D12MixedDiscretizer.binIndex(cuts, value);
                        if ((int) after.value(a) != expectedBin) throw new IllegalStateException(split + " assigned an incorrect bin for " + name + " at row " + r);
                    }
                }
            }
        }
    }

    private static List<String[]> cutPointRows(D12MixedDiscretizer discretizer) {
        List<String[]> rows = new ArrayList<>();
        rows.add(new String[]{"feature", "strategy", "requested_bins", "actual_bins", "cut_points"});
        for (Map.Entry<String, D12MixedDiscretizer.FeatureSpec> entry : D12MixedDiscretizer.config().entrySet()) {
            double[] cuts = discretizer.cutPoints().get(entry.getKey());
            rows.add(new String[]{entry.getKey(), entry.getValue().strategy().name().toLowerCase(),
                    Integer.toString(entry.getValue().requestedBins()), Integer.toString(cuts.length + 1), formatCuts(cuts)});
        }
        return rows;
    }

    private static List<String[]> binSummaryRows(D12MixedDiscretizer discretizer, Map<String, Instances> data) {
        List<String[]> rows = new ArrayList<>();
        rows.add(new String[]{"feature", "strategy", "requested_bins", "actual_bins", "cut_points", "bin_id", "interval", "display_name",
                "count_train", "share_train_pct", "count_validation", "share_validation_pct", "count_test", "share_test_pct"});
        for (Map.Entry<String, D12MixedDiscretizer.FeatureSpec> entry : D12MixedDiscretizer.config().entrySet()) {
            String feature = entry.getKey();
            double[] cuts = discretizer.cutPoints().get(feature);
            Map<String, long[]> counts = countBins(data, feature, cuts);
            for (int bin = 0; bin <= cuts.length; bin++) {
                String binId = "bin_" + bin;
                long trainCount = counts.get("Train")[bin];
                long valCount = counts.get("Validation")[bin];
                long testCount = counts.get("Test")[bin];
                rows.add(new String[]{feature, entry.getValue().strategy().name().toLowerCase(),
                        Integer.toString(entry.getValue().requestedBins()), Integer.toString(cuts.length + 1), formatCuts(cuts), binId,
                        interval(cuts, bin), displayName(entry.getValue().unit(), cuts, bin),
                        Long.toString(trainCount), pct(trainCount, sum(counts.get("Train"))),
                        Long.toString(valCount), pct(valCount, sum(counts.get("Validation"))),
                        Long.toString(testCount), pct(testCount, sum(counts.get("Test")))});
            }
        }
        return rows;
    }

    private static Map<String, long[]> countBins(Map<String, Instances> data, String feature, double[] cuts) {
        Map<String, long[]> counts = new LinkedHashMap<>();
        for (Map.Entry<String, Instances> split : data.entrySet()) {
            long[] splitCounts = new long[cuts.length + 1];
            int index = split.getValue().attribute(feature).index();
            for (int r = 0; r < split.getValue().numInstances(); r++) {
                Instance instance = split.getValue().instance(r);
                if (!instance.isMissing(index)) splitCounts[(int) instance.value(index)]++;
            }
            counts.put(split.getKey(), splitCounts);
        }
        return counts;
    }

    private static long sum(long[] values) {
        long total = 0;
        for (long value : values) total += value;
        return total;
    }

    private static List<String[]> b1Rows(Map<String, Instances> data, String positiveLabel, D12MixedDiscretizer discretizer) {
        List<String[]> rows = new ArrayList<>();
        rows.add(new String[]{"feature", "bin_id", "interval", "display_name", "split", "count", "positive_count", "positive_rate_pct", "share_pct", "baseline_rate_pct"});
        for (String feature : D12MixedDiscretizer.config().keySet()) {
            double[] cuts = discretizer.cutPoints().get(feature);
            for (String split : List.of("Train", "Validation")) {
                Instances instances = data.get(split);
                int featureIndex = instances.attribute(feature).index();
                int classIndex = instances.classIndex();
                int posIndex = instances.classAttribute().indexOfValue(positiveLabel);
                long[] counts = new long[cuts.length + 1];
                long[] positives = new long[cuts.length + 1];
                long total = 0, totalPositive = 0;
                for (int r = 0; r < instances.numInstances(); r++) {
                    Instance row = instances.instance(r);
                    if (row.isMissing(featureIndex) || row.isMissing(classIndex)) continue;
                    int bin = (int) row.value(featureIndex);
                    counts[bin]++;
                    total++;
                    if ((int) row.classValue() == posIndex) {
                        positives[bin]++;
                        totalPositive++;
                    }
                }
                D12MixedDiscretizer.FeatureSpec spec = D12MixedDiscretizer.config().get(feature);
                for (int bin = 0; bin < counts.length; bin++) {
                    rows.add(new String[]{feature, "bin_" + bin, interval(cuts, bin), displayName(spec.unit(), cuts, bin), split,
                            Long.toString(counts[bin]), Long.toString(positives[bin]), pct(positives[bin], counts[bin]),
                            pct(counts[bin], total), pct(totalPositive, total)});
                }
            }
        }
        return rows;
    }

    private static List<String[]> splitAuditRows(Map<String, Instances> data, String positiveLabel) {
        List<String[]> rows = new ArrayList<>();
        rows.add(new String[]{"split", "rows", "attributes", "positive_label", "positive_count", "negative_count", "missing_class", "missing_d12_values"});
        for (Map.Entry<String, Instances> split : data.entrySet()) {
            Instances instances = split.getValue();
            int positiveIndex = instances.classAttribute().indexOfValue(positiveLabel);
            long positive = 0, negative = 0, missingClass = 0, missingD12 = 0;
            for (int r = 0; r < instances.numInstances(); r++) {
                Instance row = instances.instance(r);
                if (row.isMissing(instances.classIndex())) missingClass++;
                else if ((int) row.classValue() == positiveIndex) positive++;
                else negative++;
                for (String feature : D12MixedDiscretizer.config().keySet()) {
                    if (row.isMissing(instances.attribute(feature).index())) missingD12++;
                }
            }
            rows.add(new String[]{split.getKey(), Integer.toString(instances.numInstances()), Integer.toString(instances.numAttributes()), positiveLabel,
                    Long.toString(positive), Long.toString(negative), Long.toString(missingClass), Long.toString(missingD12)});
        }
        return rows;
    }

    private static List<String[]> b2StatusRows(Path inputDir) {
        Path fullPrepared = Path.of("data/processed/full_prepared.csv");
        Path manifest = Path.of("outputs/preparation/split_manifest.csv");
        boolean available = Files.isRegularFile(fullPrepared) && Files.isRegularFile(manifest);
        String status = available ? "BLOCKED_REQUIRES_ROW_LINK_VALIDATION" : "BLOCKED_MISSING_LINKAGE_FILES";
        String reason = available
                ? "Both candidate sources exist, but D12Runner does not emit B2 until encounter_id membership and row order are explicitly validated against the canonical split ARFFs."
                : "No verified B2 join was available for this run. Expected data/processed/full_prepared.csv (encounter_id + original readmitted) and outputs/preparation/split_manifest.csv (encounter_id + split). Do not join by predictor values.";
        return List.of(new String[]{"status", "required_full_prepared_csv", "required_split_manifest", "reason"},
                new String[]{status, fullPrepared.toString(), manifest.toString(), reason});
    }

    private static String interval(double[] cuts, int bin) {
        String lower = bin == 0 ? "-inf" : number(cuts[bin - 1]);
        String upper = bin == cuts.length ? "+inf" : number(cuts[bin]);
        return "[" + lower + ", " + upper + ")";
    }

    private static String displayName(String unit, double[] cuts, int bin) {
        Integer low = bin == 0 ? null : (int) Math.ceil(cuts[bin - 1]);
        Integer high = bin == cuts.length ? null : (int) Math.ceil(cuts[bin]) - 1;
        if (low == null && high == null) return "Tất cả giá trị";
        if (low == null) return "≤" + high + " " + unit;
        if (high == null) return "≥" + low + " " + unit;
        if (low.equals(high)) return low + " " + unit;
        return low + "–" + high + " " + unit;
    }

    private static String formatCuts(double[] cuts) {
        if (cuts.length == 0) return "[]";
        StringBuilder result = new StringBuilder("[");
        for (int i = 0; i < cuts.length; i++) {
            if (i > 0) result.append(';');
            result.append(number(cuts[i]));
        }
        return result.append(']').toString();
    }

    private static String number(double value) { return Double.toString(value); }
    private static String pct(long numerator, long denominator) { return denominator == 0 ? "" : String.format(java.util.Locale.ROOT, "%.6f", numerator * 100.0 / denominator); }

    private static void writeCsv(Path path, List<String[]> rows) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8)) {
            for (String[] row : rows) {
                for (int i = 0; i < row.length; i++) {
                    if (i > 0) writer.write(',');
                    writer.write(csv(row[i] == null ? "" : row[i]));
                }
                writer.newLine();
            }
        }
    }

    private static String csv(String value) {
        if (value.indexOf(',') < 0 && value.indexOf('"') < 0 && value.indexOf('\n') < 0 && value.indexOf('\r') < 0) return value;
        return "\"" + value.replace("\"", "\"\"") + "\"";
    }
}
