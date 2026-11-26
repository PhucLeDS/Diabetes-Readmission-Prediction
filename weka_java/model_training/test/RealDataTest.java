import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Map;

import benchmarking.Metrics;
import model.DataLoader;
import model.DataLoader.Data;
import model.Trainer;
import model.Trainer.Candidate;
import weka.core.Instances;

/**
 * Checks that the real ARFF files in data/weka/ are ready for the pipeline, before a long run.
 * Skips (exit 0) when the files have not been generated yet. Exit code 1 on any failure.
 * args: [data folder, default data/weka] [positive label, default readmitted]
 */
public class RealDataTest {

    interface Check { void run() throws Exception; }

    static Data d;

    public static void main(String[] args) throws Exception {
        Path dir = args.length > 0 ? Paths.get(args[0]) : Paths.get("data", "weka");
        String positive = args.length > 1 ? args[1] : "readmitted";
        if (!Files.exists(dir.resolve("train.arff"))) {
            System.out.println("SKIP  no " + dir.resolve("train.arff") + " (run notebooks/02_split_export.ipynb first)");
            return;
        }
        Map<String, Check> tests = new LinkedHashMap<>();
        tests.put("load: four files, matching headers, train_validation = train + validation", () -> d = DataLoader.load(dir, positive));
        tests.put("class: two values and no missing labels", RealDataTest::classIsBinaryAndComplete);
        tests.put("pairs: development and final files have the same columns in the same order", RealDataTest::sameColumns);
        tests.put("split: positive rate is similar in train, validation and holdout", RealDataTest::similarPositiveRates);
        tests.put("signal: NaiveBayes on train ranks validation better than chance", RealDataTest::modelLearnsSomething);

        int failed = 0;
        for (Map.Entry<String, Check> t : tests.entrySet()) {
            try {
                if (d == null && !t.getKey().startsWith("load")) throw new IllegalStateException("data did not load");
                t.getValue().run();
                System.out.println("PASS  " + t.getKey());
            } catch (Throwable e) {
                failed++;
                System.out.println("FAIL  " + t.getKey() + "\n      " + e);
            }
        }
        if (d != null) {
            System.out.printf("%ntrain=%d validation=%d train_validation=%d holdout=%d, %d predictors, positive=%s%n",
                    d.train().numInstances(), d.val().numInstances(), d.trainVal().numInstances(), d.test().numInstances(),
                    d.train().numAttributes() - 1, d.positive());
        }
        System.out.printf("%d passed, %d failed%n", tests.size() - failed, failed);
        System.exit(failed == 0 ? 0 : 1);
    }

    static void classIsBinaryAndComplete() {
        for (Instances s : new Instances[]{d.train(), d.val(), d.trainVal(), d.test()}) {
            check(s.classAttribute().numValues() == 2, s.relationName() + ": class has " + s.classAttribute().numValues() + " values");
            check(s.attributeStats(s.classIndex()).missingCount == 0, s.relationName() + ": rows with a missing class");
        }
    }

    static void sameColumns() {
        check(d.train().numAttributes() == d.trainVal().numAttributes(), "column count differs between the pairs");
        for (int i = 0; i < d.train().numAttributes(); i++) {
            String a = d.train().attribute(i).name(), b = d.trainVal().attribute(i).name();
            check(a.equals(b), "column " + i + ": " + a + " vs " + b);
        }
    }

    static void similarPositiveRates() {
        double train = Metrics.baseRate(d.train(), d.pos());
        for (Instances s : new Instances[]{d.val(), d.test()}) {
            double rate = Metrics.baseRate(s, d.pos());
            check(Math.abs(rate - train) < 0.01, s.relationName() + ": positive rate " + rate + " vs train " + train);
        }
    }

    static void modelLearnsSomething() throws Exception {
        Candidate nb = Trainer.models().get("NaiveBayes").get(0);  // seconds, even on the full data
        double roc = Metrics.predict(Trainer.train(nb, d.train()), d.val()).areaUnderROC(d.pos());
        check(roc > 0.55, "validation ROC " + roc + " is close to chance");
    }

    static void check(boolean ok, String what) {
        if (!ok) throw new AssertionError(what);
    }
}
