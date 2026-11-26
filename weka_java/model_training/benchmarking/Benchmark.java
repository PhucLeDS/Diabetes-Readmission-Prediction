package benchmarking;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import benchmarking.Metrics.Scores;
import benchmarking.Tuner.Tuned;
import model.DataLoader;
import model.DataLoader.Data;
import model.Trainer;
import model.Trainer.Candidate;
import weka.classifiers.Classifier;

/**
 * Entry point. Runs load -> encode -> train -> tune -> evaluate for every model on one data
 * folder and writes outputs/tables/model_training/<folder>_results.csv.
 * See weka_java/model_training/README.md for how to build and run.
 */
public final class Benchmark {

    static final Path DEFAULT_DATA = Paths.get("data", "weka");
    static final String DEFAULT_POSITIVE = "readmitted";  // config/target_mapping.json positive_class
    static final Path OUT_DIR = Paths.get("outputs", "tables", "model_training");
    static final String HEADER = "target,dataset,model,params,threshold,"
            + "val_acc,val_roc,val_pr,val_prec,val_rec,val_f1,val_brier,"
            + "test_acc,test_roc,test_pr,test_prec,test_rec,test_f1,test_brier,test_base_rate";

    private Benchmark() {}

    /** args: [data folder, default data/weka] [positive label, default readmitted] */
    public static void main(String[] args) throws Exception {
        Path dir = args.length > 0 ? Paths.get(args[0]) : DEFAULT_DATA;
        run(dir, args.length > 1 ? args[1] : DEFAULT_POSITIVE, OUT_DIR);
    }

    /** The whole pipeline for one data folder; writes and returns <outDir>/<folder>_results.csv. */
    public static Path run(Path dir, String positive, Path outDir) throws Exception {
        Data d = DataLoader.load(dir, positive);
        System.out.printf("%s: train=%d val=%d train_validation=%d test=%d, positive=%s%n", d.name(), d.train().numInstances(),
                d.val().numInstances(), d.trainVal().numInstances(), d.test().numInstances(), d.positive());

        List<String> rows = new ArrayList<>();
        rows.add(HEADER);
        for (Map.Entry<String, List<Candidate>> m : Trainer.models().entrySet()) {
            Tuned t = Tuner.tune(d, m.getValue());
            Scores s = evaluate(t, d);
            System.out.printf("  TEST %-12s %-24s thr=%.3f acc=%.4f roc=%.4f pr=%.4f prec=%.4f rec=%.4f f1=%.4f brier=%.4f%n",
                    m.getKey(), t.cand().params(), t.threshold(), s.accuracy(), s.roc(), s.pr(), s.prec(), s.rec(), s.f1(), s.brier());
            rows.add(String.join(",", d.positive(), d.name(), m.getKey(), t.cand().params(),
                    f(t.threshold()), cells(t.val()), cells(s), f(Metrics.baseRate(d.test(), d.pos()))));
        }
        Files.createDirectories(outDir);
        Path out = outDir.resolve(d.name() + "_results.csv");
        Files.write(out, rows);
        System.out.println("saved " + out);
        return out;
    }

    /**
     * Step 5 - EVALUATE: refit the chosen setting on train + validation, then score the holdout
     * once at the threshold chosen on validation. Only trainVal and test are used here.
     */
    public static Scores evaluate(Tuned t, Data d) throws Exception {
        Classifier finalModel = Trainer.train(t.cand(), d.trainVal());
        return Metrics.score(Metrics.predict(finalModel, d.test()), d.pos(), t.threshold());
    }

    /** One split's scores as CSV cells, in HEADER order. */
    static String cells(Scores s) {
        return String.join(",", f(s.accuracy()), f(s.roc()), f(s.pr()), f(s.prec()), f(s.rec()), f(s.f1()), f(s.brier()));
    }

    /** CSV number: 4 decimals, '.' as the decimal mark on every machine. */
    static String f(double x) {
        return String.format(Locale.ROOT, "%.4f", x);
    }
}
