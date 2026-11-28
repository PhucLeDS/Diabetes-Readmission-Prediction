package weka_java.data_mining;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.SymmetricalUncertAttributeEval;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 * Feature selection F4: Symmetrical Uncertainty ranking, top 20 by default.
 *
 * <p>Self-contained: takes plain {@link Instances} with its class index already set, so it does
 * not depend on how the data was loaded. Wire it to {@code model.DataLoader.Data} from the
 * caller, for example:
 * <pre>{@code
 *     DataLoader.Data data = DataLoader.load(dir, "readmitted");
 *
 *     // Development: tune on Train, compare on Validation.
 *     F4SymmetricalUncertaintySelector devSelector = new F4SymmetricalUncertaintySelector();
 *     devSelector.fit(data.train());
 *     Instances valSelected = devSelector.transform(data.val());
 *
 *     // Final: refit on Train+Validation, score once on the held-out Test.
 *     F4SymmetricalUncertaintySelector finalSelector = new F4SymmetricalUncertaintySelector();
 *     finalSelector.fit(data.trainVal());
 *     Instances testSelected = finalSelector.transform(data.test());
 * }</pre>
 * Scoring, discretisation edges, and the selected set are learned only from the split passed to
 * {@link #fit}; {@link #transform} reuses them without refitting or rescoring.
 */
public final class F4SymmetricalUncertaintySelector {

    public static final int DEFAULT_TOP_K = 20;
    public static final int NUMERIC_BINS = 10;

    /** One feature's Symmetrical Uncertainty score and its rank (1 = best). */
    public static final class FeatureScore {
        public final String feature;
        public final double score;
        public final int rank;
        public final boolean selected;

        FeatureScore(String feature, double score, int rank, boolean selected) {
            this.feature = feature;
            this.score = score;
            this.rank = rank;
            this.selected = selected;
        }
    }

    private final int topK;
    private final List<FeatureScore> ranking = new ArrayList<>();
    private final List<String> selectedFeatures = new ArrayList<>();
    private Remove keepSelected;

    public F4SymmetricalUncertaintySelector(int topK) {
        if (topK < 1) {
            throw new IllegalArgumentException("topK must be >= 1");
        }
        this.topK = topK;
    }

    public F4SymmetricalUncertaintySelector() {
        this(DEFAULT_TOP_K);
    }

    /** Scores features on {@code train} and records the selected set. */
    public void fit(Instances train) throws Exception {
        if (train.classIndex() < 0) {
            throw new IllegalArgumentException("Train must have its class index set.");
        }
        requireNoMissing(train, numericNames(train));

        ReplaceMissingValues imputer = new ReplaceMissingValues();
        imputer.setInputFormat(train);
        Instances imputed = Filter.useFilter(train, imputer);

        Discretize discretize = new Discretize();
        discretize.setBins(NUMERIC_BINS);
        discretize.setUseEqualFrequency(true);
        discretize.setIgnoreClass(true);
        discretize.setInputFormat(imputed);
        Instances discrete = Filter.useFilter(imputed, discretize);

        AttributeSelection search = new AttributeSelection();
        search.setEvaluator(new SymmetricalUncertAttributeEval());
        search.setSearch(new Ranker());
        search.SelectAttributes(discrete);
        double[][] ranked = search.rankedAttributes(); // rows: [attribute index, merit]

        int classIndex = train.classIndex();
        int featureCount = train.numAttributes() - 1;
        if (ranked.length != featureCount) {
            throw new IllegalStateException("Expected " + featureCount + " scores, got " + ranked.length);
        }

        // Scores and names in original column order, excluding the class.
        double[] scores = new double[featureCount];
        String[] names = new String[featureCount];
        int position = 0;
        for (int i = 0; i < train.numAttributes(); i++) {
            if (i == classIndex) {
                continue;
            }
            names[position] = train.attribute(i).name();
            scores[position] = scoreOf(ranked, i);
            position++;
        }

        int[] order = orderByScore(scores);
        int[] selected = topKIndices(scores, topK);
        boolean[] isSelected = new boolean[featureCount];
        for (int index : selected) {
            isSelected[index] = true;
        }

        ranking.clear();
        selectedFeatures.clear();
        for (int rank = 0; rank < order.length; rank++) {
            int index = order[rank];
            ranking.add(new FeatureScore(names[index], scores[index], rank + 1, isSelected[index]));
        }
        for (int index : selected) {
            selectedFeatures.add(names[index]);
        }

        keepSelected = new Remove();
        keepSelected.setAttributeIndices(indicesToKeepComplement(train, selectedFeatures, classIndex));
        keepSelected.setInputFormat(train);
    }

    /** Returns the original (undiscretised) selected columns, plus the class. */
    public Instances transform(Instances data) throws Exception {
        if (keepSelected == null) {
            throw new IllegalStateException("Call fit() before transform().");
        }
        return Filter.useFilter(data, keepSelected);
    }

    public List<FeatureScore> ranking() {
        return new ArrayList<>(ranking);
    }

    public List<String> selectedFeatures() {
        return new ArrayList<>(selectedFeatures);
    }

    private static double scoreOf(double[][] ranked, int attributeIndex) {
        for (double[] row : ranked) {
            if ((int) row[0] == attributeIndex) {
                return row[1];
            }
        }
        throw new IllegalStateException("No score for attribute index " + attributeIndex);
    }

    /** Names of every non-class numeric attribute. */
    private static String[] numericNames(Instances data) {
        List<String> names = new ArrayList<>();
        for (int i = 0; i < data.numAttributes(); i++) {
            if (i != data.classIndex() && data.attribute(i).isNumeric()) {
                names.add(data.attribute(i).name());
            }
        }
        return names.toArray(new String[0]);
    }

    /** Fails fast if any of the named attributes has a missing value in {@code data}. */
    private static void requireNoMissing(Instances data, String[] names) {
        for (String name : names) {
            Attribute attribute = data.attribute(name);
            if (attribute == null) {
                throw new IllegalArgumentException("Missing attribute: " + name);
            }
            for (int i = 0; i < data.numInstances(); i++) {
                if (data.instance(i).isMissing(attribute.index())) {
                    throw new IllegalStateException("Missing value in '" + name + "' at row " + i);
                }
            }
        }
    }

    /**
     * Column indices sorted by descending score; equal scores keep the original column order.
     * Package-private (not private) so a test in this package can exercise tie-breaking directly.
     */
    static int[] orderByScore(double[] scores) {
        Integer[] order = new Integer[scores.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> {
            int byScore = Double.compare(scores[b], scores[a]);
            return byScore != 0 ? byScore : Integer.compare(a, b);
        });
        int[] result = new int[order.length];
        for (int i = 0; i < order.length; i++) {
            result[i] = order[i];
        }
        return result;
    }

    /** Indices of the k best columns, returned in ascending column order. */
    static int[] topKIndices(double[] scores, int k) {
        int count = Math.min(k, scores.length);
        int[] ordered = orderByScore(scores);
        int[] selected = Arrays.copyOf(ordered, count);
        Arrays.sort(selected);
        return selected;
    }

    /** 1-based indices of all non-class attributes that are not selected (for Weka's Remove). */
    private static String indicesToKeepComplement(Instances data, List<String> keep, int classIndex) {
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attribute = data.attribute(i);
            if (i != classIndex && !keep.contains(attribute.name())) {
                if (builder.length() > 0) {
                    builder.append(',');
                }
                builder.append(i + 1);
            }
        }
        return builder.toString();
    }
}
