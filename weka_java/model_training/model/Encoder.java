package model;

import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/** Step 2 - ENCODE: turn the data into the form a model needs. */
public final class Encoder {

    private Encoder() {}

    /**
     * For models that need numbers (Logistic): fill missing values and turn each nominal column
     * into 0/1 columns. The filters sit inside the model, so they are fitted on train only.
     * Trees and NaiveBayes read nominal columns and missing values directly: returned unchanged.
     */
    public static Classifier encode(Classifier model, boolean oneHot) {
        if (!oneHot) return model;
        MultiFilter filters = new MultiFilter();
        filters.setFilters(new Filter[]{new ReplaceMissingValues(), new NominalToBinary()});
        FilteredClassifier fc = new FilteredClassifier();
        fc.setFilter(filters);
        fc.setClassifier(model);
        return fc;
    }
}
