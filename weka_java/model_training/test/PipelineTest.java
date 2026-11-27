import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Stream;

import benchmarking.Benchmark;
import benchmarking.Metrics;
import benchmarking.Metrics.Scores;
import benchmarking.Tuner;
import benchmarking.Tuner.Tuned;
import model.DataLoader;
import model.DataLoader.Data;
import model.Encoder;
import model.Trainer;
import model.Trainer.Candidate;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

/**
 * Unit tests for the whole pipeline: load, encode, train, tune, evaluate, metrics, and one full run.
 * Plain Java, no test library: prints PASS/FAIL per test and exits with code 1 on any failure.
 * Uses small synthetic data in a temp folder, so it never reads or writes project data.
 * RealDataTest checks the real data/weka files. See weka_java/model_training/README.md.
 */
public class PipelineTest {

    static final String POS = "readmitted";
    static Path tmp;

    interface Check { void run() throws Exception; }

    public static void main(String[] args) throws Exception {
        tmp = Files.createTempDirectory("weka_model_training_test");
        Map<String, Check> tests = new LinkedHashMap<>();
        tests.put("metrics: known confusion, accuracy, precision, recall, F1, Brier, ROC", PipelineTest::metricsKnownValues);
        tests.put("metrics: best-F1 threshold is the expected one", PipelineTest::bestThreshold);
        tests.put("metrics: F1 is 0 when nothing is predicted positive", PipelineTest::f1ZeroWhenNoPositives);
        tests.put("load: reads the four files and finds the positive label", PipelineTest::loadReadsFiles);
        tests.put("load: unknown positive label is rejected", PipelineTest::loadRejectsUnknownPositive);
        tests.put("load: a pair with different columns is rejected", PipelineTest::loadRejectsMismatchedHeader);
        tests.put("load: train_validation must be train + validation", PipelineTest::loadRejectsWrongTrainValidation);
        tests.put("load: missing file is rejected", PipelineTest::loadRejectsMissingFile);
        tests.put("encode: one-hot wrapper only when asked", PipelineTest::encodeWrapsOnlyWhenOneHot);
        tests.put("train: every model setting gives valid probabilities", PipelineTest::everyModelPredicts);
        tests.put("train: class balancing does not change the caller's train data", PipelineTest::balancingLeavesTrainUntouched);
        tests.put("tune: uses only train and validation", PipelineTest::tuneUsesOnlyTrainAndValidation);
        tests.put("tune: keeps the setting with the best validation PR-AUC", PipelineTest::tuneKeepsBestPr);
        tests.put("evaluate: uses only train_validation and the holdout", PipelineTest::evaluateUsesOnlyFinalPair);
        tests.put("benchmark: full run writes one valid CSV row per model", PipelineTest::benchmarkEndToEnd);

        int failed = 0;
        for (Map.Entry<String, Check> t : tests.entrySet()) {
            try {
                t.getValue().run();
                System.out.println("PASS  " + t.getKey());
            } catch (Throwable e) {
                failed++;
                System.out.println("FAIL  " + t.getKey() + "\n      " + e);
            }
        }
        deleteTree(tmp);
        System.out.printf("%n%d passed, %d failed%n", tests.size() - failed, failed);
        System.exit(failed == 0 ? 0 : 1);
    }

    // ---------- metrics ----------

    /** 6 rows with known p(positive): tp=2 fp=1 fn=1 tn=2 at threshold 0.5. */
    static Evaluation knownPredictions() throws Exception {
        double[][] rows = {{0.9, 1}, {0.8, 1}, {0.7, 0}, {0.4, 1}, {0.2, 0}, {0.1, 0}};
        ArrayList<Attribute> atts = new ArrayList<>(List.of(new Attribute("p"), new Attribute("class", List.of("not_readmitted", POS))));
        Instances d = new Instances("known", atts, rows.length);
        d.setClassIndex(1);
        for (double[] r : rows) d.add(new DenseInstance(1.0, r));
        return Metrics.predict(new FixedProbability(), d);
    }

    static void metricsKnownValues() throws Exception {
        Evaluation ev = knownPredictions();
        int[] c = Metrics.confusion(ev, 1, 0.5);
        check(c[0] == 2 && c[1] == 1 && c[2] == 1 && c[3] == 2, "confusion {tp,fp,fn,tn} = " + Arrays.toString(c));
        Scores s = Metrics.score(ev, 1, 0.5);
        close(s.accuracy(), 4.0 / 6, "accuracy");
        close(s.prec(), 2.0 / 3, "precision");
        close(s.rec(), 2.0 / 3, "recall");
        close(s.f1(), 2.0 / 3, "f1");
        close(s.brier(), 0.95 / 6, "brier");
        close(s.roc(), 8.0 / 9, "roc");  // 8 of the 9 positive/negative pairs are ranked correctly
        check(s.pr() > 0 && s.pr() <= 1, "pr-auc in (0,1]: " + s.pr());
    }

    static void bestThreshold() throws Exception {
        // F1 peaks (6/7) when the 4 highest are positive: first threshold above 0.2 is 0.205
        close(Metrics.bestF1Threshold(knownPredictions(), 1), 0.205, "best threshold");
    }

    static void f1ZeroWhenNoPositives() {
        close(Metrics.f1(new int[]{0, 0, 3, 3}), 0, "f1 with no positives predicted");
    }

    /** Returns the row's first column as p(positive), so test predictions are known in advance. */
    static class FixedProbability extends AbstractClassifier {
        @Override public void buildClassifier(Instances data) {}
        @Override public double[] distributionForInstance(Instance x) { return new double[]{1 - x.value(0), x.value(0)}; }
    }

    // ---------- load ----------

    static void loadReadsFiles() throws Exception {
        Data d = DataLoader.load(dataset("T1_LOAD"), POS);
        check(d.train().numInstances() == 600 && d.val().numInstances() == 300, "train/validation sizes");
        check(d.trainVal().numInstances() == 900 && d.test().numInstances() == 300, "train_validation/test sizes");
        check(d.positive().equals(POS) && d.pos() == 1, "positive label index");
        check(d.train().classIndex() == d.train().numAttributes() - 1, "class is the last column");
        check(d.name().equals("T1_LOAD"), "name is the folder name: " + d.name());
    }

    static void loadRejectsUnknownPositive() throws Exception {
        Path dir = dataset("T2_LABEL");
        throwsType(IllegalArgumentException.class, () -> DataLoader.load(dir, "readmitted_lt30"));
    }

    static void loadRejectsMismatchedHeader() throws Exception {
        Instances train = synthetic(600, 1), val = synthetic(300, 2);
        Instances test = synthetic(300, 3);
        test.setClassIndex(-1);
        test.deleteAttributeAt(1);  // holdout loses column "x"
        Path dir = write("T3_HEADER", train, val, concat(train, val), test);
        throwsType(IllegalArgumentException.class, () -> DataLoader.load(dir, POS));
    }

    static void loadRejectsWrongTrainValidation() throws Exception {
        Instances train = synthetic(600, 1), val = synthetic(300, 2);
        Path dir = write("T4_TRAINVAL", train, val, train, synthetic(300, 3));  // train only, validation missing
        throwsType(IllegalArgumentException.class, () -> DataLoader.load(dir, POS));
    }

    static void loadRejectsMissingFile() throws Exception {
        Path dir = dataset("T5_MISSING");
        Files.delete(dir.resolve("test_for_train_validation.arff"));
        throwsType(IOException.class, () -> DataLoader.load(dir, POS));
    }

    // ---------- encode ----------

    static void encodeWrapsOnlyWhenOneHot() {
        J48 tree = new J48();
        check(Encoder.encode(tree, false) == tree, "no one-hot: model returned unchanged");
        Classifier wrapped = Encoder.encode(new Logistic(), true);
        check(wrapped instanceof FilteredClassifier fc && fc.getClassifier() instanceof Logistic, "one-hot: Logistic inside a FilteredClassifier");
    }

    // ---------- train ----------

    static void everyModelPredicts() throws Exception {
        Data d = DataLoader.load(dataset("T6_TRAIN"), POS);
        for (List<Candidate> grid : Trainer.models().values()) {
            for (Candidate c : grid) {
                Classifier m = Trainer.train(c, d.train());
                for (int i = 0; i < d.val().numInstances(); i++) {
                    double[] p = m.distributionForInstance(d.val().instance(i));
                    // Weka can return -1e-19 for a zero probability: allow rounding noise
                    check(p.length == 2 && p[1] > -1e-9 && p[1] < 1 + 1e-9 && Math.abs(p[0] + p[1] - 1) < 1e-6,
                            c.model() + " " + c.params() + ": bad distribution " + Arrays.toString(p));
                }
            }
        }
    }

    static void balancingLeavesTrainUntouched() throws Exception {
        Data d = DataLoader.load(dataset("T7_BALANCE"), POS);
        Candidate balanced = Trainer.models().get("J48").get(0);
        check(balanced.balance(), "J48 is configured with class balancing");
        Trainer.train(balanced, d.train());
        for (int i = 0; i < d.train().numInstances(); i++) {
            check(d.train().instance(i).weight() == 1.0, "train row " + i + " weight changed to " + d.train().instance(i).weight());
        }
    }

    // ---------- tune ----------

    static void tuneUsesOnlyTrainAndValidation() throws Exception {
        Data full = DataLoader.load(dataset("T8_TUNE"), POS);
        Data devOnly = new Data(full.name(), full.train(), full.val(), null, null, full.pos());  // any other use throws
        Tuned t = Tuner.tune(devOnly, Trainer.models().get("Logistic"));
        check(Trainer.models().get("Logistic").stream().anyMatch(c -> c.params().equals(t.cand().params())), "picked a grid setting");
        check(t.threshold() > 0 && t.threshold() < 1, "threshold in (0,1): " + t.threshold());
    }

    static void tuneKeepsBestPr() throws Exception {
        Data d = DataLoader.load(dataset("T9_BEST"), POS);
        List<Candidate> grid = Trainer.models().get("LogitBoost");
        Tuned t = Tuner.tune(d, grid);
        double bestPr = 0;
        for (Candidate c : grid) {
            Evaluation ev = Metrics.predict(Trainer.train(c, d.train()), d.val());
            bestPr = Math.max(bestPr, ev.areaUnderPRC(d.pos()));
        }
        close(t.val().pr(), bestPr, "tuned validation PR-AUC equals the best in the grid");
    }

    // ---------- evaluate ----------

    static void evaluateUsesOnlyFinalPair() throws Exception {
        Data full = DataLoader.load(dataset("T10_EVAL"), POS);
        Tuned t = Tuner.tune(full, Trainer.models().get("Logistic"));
        Data finalOnly = new Data(full.name(), null, null, full.trainVal(), full.test(), full.pos());  // any other use throws
        Scores s = Benchmark.evaluate(t, finalOnly);
        check(s.roc() > 0.7, "refit model learns the synthetic signal, holdout ROC = " + s.roc());
    }

    // ---------- whole pipeline ----------

    static void benchmarkEndToEnd() throws Exception {
        Path out = Benchmark.run(dataset("T11_FULL"), POS, tmp.resolve("out"));
        check(out.getFileName().toString().equals("T11_FULL_results.csv"), "output file name: " + out.getFileName());
        List<String> lines = Files.readAllLines(out);
        check(lines.size() == 1 + Trainer.models().size(), "header + one row per model, got " + lines.size());
        List<String> header = List.of(lines.get(0).split(","));
        for (String row : lines.subList(1, lines.size())) {
            String[] cells = row.split(",");
            check(cells.length == header.size(), "row has " + cells.length + " cells, header has " + header.size());
            for (int i = header.indexOf("threshold"); i < cells.length; i++) {
                double v = Double.parseDouble(cells[i]);
                check(v >= 0 && v <= 1, header.get(i) + " out of [0,1]: " + v);
            }
        }
        String logistic = lines.stream().filter(l -> l.contains(",Logistic,")).findFirst().orElseThrow();
        double roc = Double.parseDouble(logistic.split(",")[header.indexOf("test_roc")]);
        check(roc > 0.7, "Logistic should learn the synthetic signal, test ROC = " + roc);
    }

    // ---------- helpers ----------

    /** Synthetic rows: nominal 'color' (some missing), numeric 'x', class depends on both (~25% positive). */
    static Instances synthetic(int n, long seed) {
        ArrayList<Attribute> atts = new ArrayList<>(List.of(
                new Attribute("color", List.of("a", "b", "c")), new Attribute("x"),
                new Attribute("class", List.of("not_readmitted", POS))));
        Instances d = new Instances("synthetic", atts, n);
        d.setClassIndex(2);
        Random r = new Random(seed);
        for (int i = 0; i < n; i++) {
            int color = r.nextInt(3);
            double x = r.nextGaussian();
            double logit = -1.6 + 1.5 * x + (color == 2 ? 1.0 : 0.0);
            double y = r.nextDouble() < 1 / (1 + Math.exp(-logit)) ? 1 : 0;
            d.add(new DenseInstance(1.0, new double[]{i % 25 == 0 ? Utils.missingValue() : color, x, y}));
        }
        return d;
    }

    /** Rows of `a` followed by rows of `b` (same header), like train_validation.arff. */
    static Instances concat(Instances a, Instances b) {
        Instances out = new Instances(a);
        for (int i = 0; i < b.numInstances(); i++) out.add(b.instance(i));
        return out;
    }

    /** A valid data folder named `name`: 600 train, 300 validation, 900 train_validation, 300 test rows. */
    static Path dataset(String name) throws IOException {
        Instances train = synthetic(600, 1), val = synthetic(300, 2);
        return write(name, train, val, concat(train, val), synthetic(300, 3));
    }

    /** Write the four files under <tmp>/<name>/, the layout DataLoader expects (as in data/weka/). */
    static Path write(String name, Instances train, Instances val, Instances trainVal, Instances test) throws IOException {
        Path dir = Files.createDirectories(tmp.resolve(name));
        Files.writeString(dir.resolve("train.arff"), train.toString());
        Files.writeString(dir.resolve("validation.arff"), val.toString());
        Files.writeString(dir.resolve("train_validation.arff"), trainVal.toString());
        Files.writeString(dir.resolve("test_for_train_validation.arff"), test.toString());
        return dir;
    }

    static void check(boolean ok, String what) {
        if (!ok) throw new AssertionError(what);
    }

    static void close(double actual, double expected, String what) {
        check(Math.abs(actual - expected) < 1e-9, what + ": expected " + expected + ", got " + actual);
    }

    static void throwsType(Class<? extends Throwable> type, Check c) {
        try {
            c.run();
        } catch (Throwable e) {
            check(type.isInstance(e), "expected " + type.getSimpleName() + ", got " + e);
            return;
        }
        throw new AssertionError("expected " + type.getSimpleName() + ", nothing was thrown");
    }

    static void deleteTree(Path root) throws IOException {
        try (Stream<Path> paths = Files.walk(root)) {
            paths.sorted(Comparator.reverseOrder()).forEach(p -> p.toFile().delete());
        }
    }
}
