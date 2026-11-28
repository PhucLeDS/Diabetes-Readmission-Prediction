package diabetes.datamining;

import java.util.ArrayList;
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

/** Feature selection F4: Symmetrical Uncertainty ranking, top 20 by default.*/
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

    /** Scores features on Train and records the selected set. Validation and test are not used. */
    public void fit(Instances train) throws Exception {
        if (train.classIndex() < 0) {
            throw new IllegalArgumentException("Train must have its class index set.");
        }
        D0Loader.requireNoMissing(train, D0Loader.numericNames(train));

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

        // Scores in original column order, excluding the class.
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

        int[] order = ColumnPolicy.orderByScore(scores);
        int[] selected = ColumnPolicy.topKIndices(scores, topK);
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
        keepSelected.setAttributeIndices(indicesToRemove(train, selectedFeatures, classIndex));
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

    /** 1-based indices of all non-class attributes that are not selected. */
    private static String indicesToRemove(Instances data, List<String> keep, int classIndex) {
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
