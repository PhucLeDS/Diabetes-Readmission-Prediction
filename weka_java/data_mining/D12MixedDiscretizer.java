import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/** D12 mixed discretization. Cut points are fitted from Train only and do not use the class. */
public final class D12MixedDiscretizer {

    public enum Strategy { DOMAIN_BASED, EQUAL_FREQUENCY, OPTIMIZED_BINS }

    public record FeatureSpec(Strategy strategy, int requestedBins, double[] domainCuts, String unit) {
        public FeatureSpec {
            domainCuts = domainCuts == null ? null : domainCuts.clone();
        }
        @Override public double[] domainCuts() { return domainCuts == null ? null : domainCuts.clone(); }
    }

    private static final Map<String, FeatureSpec> CONFIG;
    static {
        Map<String, FeatureSpec> c = new LinkedHashMap<>();
        c.put("time_in_hospital", new FeatureSpec(Strategy.DOMAIN_BASED, 4, new double[]{2.5, 5.5, 9.5}, "ngày"));
        c.put("num_lab_procedures", new FeatureSpec(Strategy.EQUAL_FREQUENCY, 5, null, "xét nghiệm"));
        c.put("num_procedures", new FeatureSpec(Strategy.DOMAIN_BASED, 4, new double[]{0.5, 1.5, 2.5}, "thủ thuật"));
        c.put("num_medications", new FeatureSpec(Strategy.EQUAL_FREQUENCY, 5, null, "thuốc"));
        c.put("number_outpatient", new FeatureSpec(Strategy.DOMAIN_BASED, 4, new double[]{0.5, 1.5, 2.5}, "lần ngoại trú"));
        c.put("number_emergency", new FeatureSpec(Strategy.DOMAIN_BASED, 4, new double[]{0.5, 1.5, 2.5}, "lần cấp cứu"));
        c.put("number_inpatient", new FeatureSpec(Strategy.DOMAIN_BASED, 4, new double[]{0.5, 1.5, 2.5}, "lần nhập viện"));
        c.put("number_diagnoses", new FeatureSpec(Strategy.OPTIMIZED_BINS, 10, null, "chẩn đoán"));
        CONFIG = Collections.unmodifiableMap(c);
    }

    private final Map<String, double[]> cutPoints = new LinkedHashMap<>();
    private Instances trainHeader;
    private boolean fitted;

    public static Map<String, FeatureSpec> config() { return CONFIG; }

    public D12MixedDiscretizer fit(Instances train) {
        cutPoints.clear();
        fitted = false;
        validateClass(train, "Train");
        for (Map.Entry<String, FeatureSpec> entry : CONFIG.entrySet()) {
            String name = entry.getKey();
            int index = train.attribute(name) == null ? -1 : train.attribute(name).index();
            if (index < 0) throw new IllegalArgumentException("Train is missing required predictor '" + name + "'");
            Attribute attribute = train.attribute(index);
            if (!attribute.isNumeric()) throw new IllegalArgumentException("Train predictor '" + name + "' must be numeric, found " + attribute.type());
            double[] values = observedFiniteValues(train, index, name, "Train");
            FeatureSpec spec = entry.getValue();
            double[] cuts = switch (spec.strategy()) {
                case DOMAIN_BASED -> spec.domainCuts();
                case EQUAL_FREQUENCY -> equalFrequencyCutPoints(values, spec.requestedBins());
                case OPTIMIZED_BINS -> optimizedCutPoints(values, spec.requestedBins(), 0.01, 0.02);
            };
            cutPoints.put(name, cuts);
        }
        for (int i = 0; i < train.numAttributes(); i++) {
            Attribute attribute = train.attribute(i);
            if (i != train.classIndex() && !attribute.isNumeric() && !attribute.isNominal()) {
                throw new IllegalArgumentException("Unsupported attribute type for '" + attribute.name() + "': " + attribute.type());
            }
        }
        trainHeader = new Instances(train, 0);
        fitted = true;
        return this;
    }

    public Instances transform(Instances input) {
        if (!fitted) throw new IllegalStateException("Call fit(train) before transform(data)");
        validateCompatibleHeader(trainHeader, input, "input");

        ArrayList<Attribute> outputAttributes = new ArrayList<>();
        for (int a = 0; a < input.numAttributes(); a++) {
            Attribute source = input.attribute(a);
            double[] cuts = cutPoints.get(source.name());
            if (cuts != null) {
                ArrayList<String> values = new ArrayList<>();
                for (int b = 0; b <= cuts.length; b++) values.add("bin_" + b);
                outputAttributes.add(new Attribute(source.name(), values));
            } else if (source.isNumeric()) {
                outputAttributes.add(new Attribute(source.name()));
            } else if (source.isNominal()) {
                ArrayList<String> values = new ArrayList<>();
                for (int v = 0; v < source.numValues(); v++) values.add(source.value(v));
                outputAttributes.add(new Attribute(source.name(), values));
            } else {
                throw new IllegalArgumentException("Unsupported attribute type for '" + source.name() + "': " + source.type());
            }
        }

        Instances output = new Instances(trainHeader.relationName() + "_D12", outputAttributes, input.numInstances());
        output.setClassIndex(input.classIndex());
        for (int r = 0; r < input.numInstances(); r++) {
            Instance row = input.instance(r);
            double[] values = new double[input.numAttributes()];
            Arrays.fill(values, Double.NaN); // Weka represents a missing numeric/nominal value as NaN.
            for (int a = 0; a < input.numAttributes(); a++) {
                Attribute source = input.attribute(a);
                Attribute target = output.attribute(a);
                if (row.isMissing(a)) continue;
                double[] cuts = cutPoints.get(source.name());
                if (cuts != null) {
                    double value = row.value(a);
                    if (Double.isNaN(value) || value == Double.POSITIVE_INFINITY) continue;
                    if (value == Double.NEGATIVE_INFINITY) {
                        values[a] = 0;
                    } else {
                        values[a] = binIndex(cuts, value);
                    }
                } else if (source.isNominal()) {
                    String label = row.stringValue(a);
                    int mapped = target.indexOfValue(label);
                    if (mapped < 0) throw new IllegalArgumentException("Value '" + label + "' in " + source.name() + " is absent from the Train header");
                    values[a] = mapped;
                } else {
                    values[a] = row.value(a);
                }
            }
            output.add(new DenseInstance(row.weight(), values));
        }
        return output;
    }

    public Map<String, double[]> cutPoints() {
        if (!fitted) throw new IllegalStateException("Call fit(train) first");
        Map<String, double[]> copy = new LinkedHashMap<>();
        cutPoints.forEach((name, cuts) -> copy.put(name, cuts.clone()));
        return Collections.unmodifiableMap(copy);
    }

    public static int binIndex(double[] cuts, double value) {
        int low = 0, high = cuts.length;
        while (low < high) {
            int mid = (low + high) >>> 1;
            if (value < cuts[mid]) high = mid;
            else low = mid + 1; // Equality belongs to the bin on the right: [lower, upper).
        }
        return low;
    }

    /** NumPy's default linear quantile method, with only unique interior cut points retained. */
    static double[] equalFrequencyCutPoints(double[] input, int requestedBins) {
        if (requestedBins < 2) throw new IllegalArgumentException("requestedBins must be >= 2");
        double[] sorted = finiteSorted(input, "equal_frequency");
        if (sorted.length == 0) throw new IllegalArgumentException("Cannot fit equal_frequency on no observed values");
        double min = sorted[0], max = sorted[sorted.length - 1];
        if (min == max) return new double[0];
        List<Double> cuts = new ArrayList<>();
        for (int q = 1; q < requestedBins; q++) {
            double position = (sorted.length - 1.0) * q / requestedBins;
            int lower = (int) Math.floor(position);
            int upper = (int) Math.ceil(position);
            double quantile = sorted[lower] + (position - lower) * (sorted[upper] - sorted[lower]);
            if (quantile <= min || quantile >= max) continue;
            if (cuts.isEmpty() || Double.compare(cuts.get(cuts.size() - 1), quantile) != 0) cuts.add(quantile);
        }
        return toArray(cuts);
    }

    /** Port of Discretizer._optimized_cut_points: greedy global-count gain and first-tie wins. */
    static double[] optimizedCutPoints(double[] input, int maxBins, double minBinFraction, double minGain) {
        if (maxBins < 1) throw new IllegalArgumentException("maxBins must be >= 1");
        double[] sorted = finiteSorted(input, "optimized_bins");
        if (sorted.length == 0) throw new IllegalArgumentException("Cannot fit optimized_bins on no observed values");
        double[] unique = new double[sorted.length];
        int[] counts = new int[sorted.length];
        int uniqueCount = 0;
        for (double value : sorted) {
            if (uniqueCount == 0 || value != unique[uniqueCount - 1]) {
                unique[uniqueCount] = value;
                counts[uniqueCount] = 1;
                uniqueCount++;
            } else counts[uniqueCount - 1]++;
        }
        unique = Arrays.copyOf(unique, uniqueCount);
        counts = Arrays.copyOf(counts, uniqueCount);
        long[] cumulative = new long[uniqueCount];
        for (int i = 0; i < uniqueCount; i++) cumulative[i] = counts[i] + (i == 0 ? 0 : cumulative[i - 1]);
        long total = sorted.length;
        long minCount = (long) Math.ceil(total * minBinFraction);

        List<int[]> groups = new ArrayList<>();
        groups.add(new int[]{0, uniqueCount - 1});
        while (groups.size() < maxBins) {
            int bestGroup = -1, bestBoundary = -1;
            double bestGain = minGain;
            for (int g = 0; g < groups.size(); g++) {
                int start = groups.get(g)[0], end = groups.get(g)[1];
                if (start == end) continue;
                long before = start > 0 ? cumulative[start - 1] : 0;
                long groupCount = cumulative[end] - before;
                for (int split = start; split < end; split++) {
                    long leftCount = cumulative[split] - before;
                    long rightCount = groupCount - leftCount;
                    if (Math.min(leftCount, rightCount) < minCount) continue;
                    double pLeft = (double) leftCount / total;
                    double pRight = (double) rightCount / total;
                    double gain = 2 * pLeft * pRight;
                    if (gain > bestGain) {
                        bestGain = gain;
                        bestGroup = g;
                        bestBoundary = split;
                    }
                }
            }
            if (bestGroup < 0) break;
            int[] old = groups.remove(bestGroup);
            groups.add(new int[]{old[0], bestBoundary});
            groups.add(new int[]{bestBoundary + 1, old[1]});
            groups.sort((a, b) -> Integer.compare(a[0], b[0]));
        }
        double[] cuts = new double[groups.size() - 1];
        for (int i = 0; i < cuts.length; i++) {
            double left = unique[groups.get(i)[1]], right = unique[groups.get(i + 1)[0]];
            cuts[i] = left + (right - left) / 2.0;
        }
        return cuts;
    }

    private static double[] observedFiniteValues(Instances data, int column, String name, String split) {
        double[] temp = new double[data.numInstances()];
        int n = 0;
        for (int r = 0; r < data.numInstances(); r++) {
            Instance row = data.instance(r);
            if (row.isMissing(column)) continue;
            double value = row.value(column);
            if (!Double.isFinite(value)) throw new IllegalArgumentException(split + " predictor '" + name + "' contains non-finite value at row " + r);
            temp[n++] = value;
        }
        if (n == 0) throw new IllegalArgumentException(split + " predictor '" + name + "' has no observed values");
        return Arrays.copyOf(temp, n);
    }

    private static double[] finiteSorted(double[] input, String context) {
        double[] values = input.clone();
        for (double value : values) if (!Double.isFinite(value)) throw new IllegalArgumentException(context + " values must be finite");
        Arrays.sort(values);
        return values;
    }

    private static double[] toArray(List<Double> values) {
        double[] result = new double[values.size()];
        for (int i = 0; i < result.length; i++) result[i] = values.get(i);
        return result;
    }

    private static void validateClass(Instances data, String split) {
        if (data == null) throw new IllegalArgumentException(split + " data is null");
        if (data.numAttributes() == 0) throw new IllegalArgumentException(split + " has no attributes");
        if (data.classIndex() != data.numAttributes() - 1) throw new IllegalArgumentException(split + " class must be the final attribute (set classIndex to the final column)");
        if (!data.classAttribute().isNominal()) throw new IllegalArgumentException(split + " class must be nominal");
    }

    private static void validateCompatibleHeader(Instances reference, Instances candidate, String split) {
        validateClass(candidate, split);
        if (!reference.equalHeaders(candidate)) {
            throw new IllegalArgumentException(split + " header is incompatible with Train: " + reference.equalHeadersMsg(candidate));
        }
    }
}
