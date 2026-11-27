package model;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.LogitBoost;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.ClassBalancer;

/** Step 3 - TRAIN: the Weka models, the settings to try for each, and fitting one on train. */
public final class Trainer {

    public static final int SEED = 42;

    /** One model setting: how to build it, and whether it needs one-hot encoding / class balancing. */
    public record Candidate(String model, String params, Supplier<Classifier> make, boolean oneHot, boolean balance) {}

    private Trainer() {}

    /** Every model and the settings the tuner tries for it. Add a model or a setting here. */
    public static Map<String, List<Candidate>> models() {
        Map<String, List<Candidate>> m = new LinkedHashMap<>();

        List<Candidate> lr = new ArrayList<>();
        for (double ridge : new double[]{1e-8, 1.0, 100.0}) {
            lr.add(new Candidate("Logistic", "ridge=" + ridge,
                    () -> { Logistic c = new Logistic(); c.setRidge(ridge); return c; }, true, false));
        }
        m.put("Logistic", lr);

        List<Candidate> nb = new ArrayList<>();
        for (boolean disc : new boolean[]{false, true}) {
            nb.add(new Candidate("NaiveBayes", "discretize=" + disc,
                    () -> { NaiveBayes c = new NaiveBayes(); c.setUseSupervisedDiscretization(disc); return c; }, false, false));
        }
        m.put("NaiveBayes", nb);

        // J48 and RandomForest train on class-balanced weights: unbalanced, J48 predicts "other" for everyone
        List<Candidate> j48 = new ArrayList<>();
        // big leaves + Laplace smoothing give J48 usable probabilities instead of near-0/1 votes
        for (int minObj : new int[]{50, 200, 500}) {
            j48.add(new Candidate("J48", "minNumObj=" + minObj + ";laplace",
                    () -> { J48 c = new J48(); c.setMinNumObj(minObj); c.setUseLaplace(true); return c; }, false, true));
        }
        m.put("J48", j48);

        List<Candidate> rf = new ArrayList<>();
        // minimum leaf size is what stops each tree memorising single patients on this noisy target
        for (int minLeaf : new int[]{1, 25, 100}) {
            rf.add(new Candidate("RandomForest", "trees=200;minLeaf=" + minLeaf, () -> {
                RandomForest c = new RandomForest();
                c.setNumIterations(200);
                ((RandomTree) c.getClassifier()).setMinNum(minLeaf);
                c.setSeed(SEED);
                c.setNumExecutionSlots(Runtime.getRuntime().availableProcessors());
                return c;
            }, false, true));
        }
        m.put("RandomForest", rf);

        List<Candidate> lb = new ArrayList<>();  // LogitBoost's default base learner is DecisionStump
        for (int iters : new int[]{100, 200, 500}) {
            lb.add(new Candidate("LogitBoost", "iterations=" + iters,
                    () -> { LogitBoost c = new LogitBoost(); c.setNumIterations(iters); c.setSeed(SEED); return c; }, false, false));
        }
        m.put("LogitBoost", lb);
        return m;
    }

    /** Fit one setting on the training split. Class balancing, if on, reweights a copy of train only. */
    public static Classifier train(Candidate c, Instances train) throws Exception {
        Instances data = train;
        if (c.balance()) {
            ClassBalancer balancer = new ClassBalancer();
            balancer.setInputFormat(train);
            data = Filter.useFilter(train, balancer);
        }
        Classifier model = Encoder.encode(c.make().get(), c.oneHot());
        model.buildClassifier(data);
        return model;
    }
}
