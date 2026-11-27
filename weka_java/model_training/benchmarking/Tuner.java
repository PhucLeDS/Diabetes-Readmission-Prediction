package benchmarking;

import java.util.List;

import benchmarking.Metrics.Scores;
import model.DataLoader.Data;
import model.Trainer;
import model.Trainer.Candidate;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;

/** Step 4 - TUNE: choose each model's setting and decision threshold on the validation split. */
public final class Tuner {

    /** What tune() picked for one model: the setting, the fitted model, the threshold and its validation scores. */
    public record Tuned(Candidate cand, Classifier model, double threshold, Scores val) {}

    private Tuner() {}

    /**
     * Train every setting of one model on train, keep the one with the best validation PR-AUC
     * (the useful ranking metric at ~9% positives), then pick the threshold with the best
     * validation F1. Test data is never touched here.
     */
    public static Tuned tune(Data d, List<Candidate> grid) throws Exception {
        Tuned best = null;
        for (Candidate c : grid) {
            long start = System.currentTimeMillis();
            Classifier model = Trainer.train(c, d.train());
            Evaluation ev = Metrics.predict(model, d.val());
            double threshold = Metrics.bestF1Threshold(ev, d.pos());
            Scores s = Metrics.score(ev, d.pos(), threshold);
            System.out.printf("  val  %-12s %-24s thr=%.3f roc=%.4f pr=%.4f f1=%.4f (%ds)%n", c.model(), c.params(),
                    threshold, s.roc(), s.pr(), s.f1(), (System.currentTimeMillis() - start) / 1000);
            if (best == null || s.pr() > best.val().pr()) best = new Tuned(c, model, threshold, s);
        }
        return best;
    }
}
