package benchmarking;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.classifiers.evaluation.Prediction;
import weka.core.Instances;

/**
 * Metrics for the positive class. "Predicted positive" means p(positive) >= threshold, so
 * accuracy, precision, recall and F1 all use the threshold chosen on validation;
 * ROC-AUC, PR-AUC and Brier use the probabilities and need no threshold.
 */
public final class Metrics {

    /** Scores on one split. */
    public record Scores(double accuracy, double roc, double pr, double prec, double rec, double f1, double brier) {}

    private Metrics() {}

    /** Run the model over a split once; Weka keeps every prediction for the metrics below. */
    public static Evaluation predict(Classifier model, Instances data) throws Exception {
        Evaluation ev = new Evaluation(data);
        ev.evaluateModel(model, data);
        return ev;
    }

    /**
     * All scores at one threshold. ROC-AUC and PR-AUC come from Weka (its PR area is close to,
     * but not computed exactly like, sklearn's average_precision).
     */
    public static Scores score(Evaluation ev, int pos, double threshold) {
        int[] c = confusion(ev, pos, threshold);
        int tp = c[0], fp = c[1], fn = c[2], tn = c[3];
        double prec = tp + fp == 0 ? 0 : (double) tp / (tp + fp);
        double rec = tp + fn == 0 ? 0 : (double) tp / (tp + fn);
        double accuracy = (double) (tp + tn) / (tp + fp + fn + tn);
        return new Scores(accuracy, ev.areaUnderROC(pos), ev.areaUnderPRC(pos), prec, rec, f1(c), brier(ev, pos));
    }

    /** Confusion counts {tp, fp, fn, tn} when predicting positive for p >= threshold. */
    public static int[] confusion(Evaluation ev, int pos, double threshold) {
        int tp = 0, fp = 0, fn = 0, tn = 0;
        for (Prediction p : ev.predictions()) {
            boolean predicted = ((NominalPrediction) p).distribution()[pos] >= threshold;
            boolean actual = (int) p.actual() == pos;
            if (predicted && actual) tp++;
            else if (predicted) fp++;
            else if (actual) fn++;
            else tn++;
        }
        return new int[]{tp, fp, fn, tn};
    }

    /** F1 of the positive class from confusion counts; 0 when nothing is predicted or present. */
    public static double f1(int[] c) {
        int tp = c[0], fp = c[1], fn = c[2];
        return tp == 0 ? 0 : 2.0 * tp / (2.0 * tp + fp + fn);
    }

    /** Brier score: mean of (p(positive) - actual)^2, actual = 1 for positive rows. Lower is better. */
    public static double brier(Evaluation ev, int pos) {
        double sum = 0;
        for (Prediction p : ev.predictions()) {
            double y = (int) p.actual() == pos ? 1 : 0;
            double d = ((NominalPrediction) p).distribution()[pos] - y;
            sum += d * d;
        }
        return sum / ev.predictions().size();
    }

    /** Threshold (0.005 to 0.995 in steps of 0.005) with the highest F1 on the given predictions. */
    public static double bestF1Threshold(Evaluation ev, int pos) {
        double bestT = 0.5, bestF1 = -1;
        for (int i = 1; i < 200; i++) {
            double t = i / 200.0;
            double f1 = f1(confusion(ev, pos, t));
            if (f1 > bestF1) { bestF1 = f1; bestT = t; }
        }
        return bestT;
    }

    /** Share of positive rows in a split. */
    public static double baseRate(Instances data, int pos) {
        int n = 0;
        for (int i = 0; i < data.numInstances(); i++) if ((int) data.instance(i).classValue() == pos) n++;
        return (double) n / data.numInstances();
    }
}
