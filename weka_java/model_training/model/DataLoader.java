package model;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * Step 1 - LOAD: read the ARFF files written by notebooks/02_split_export.ipynb (data/weka/).
 *
 *   train.arff + validation.arff                          development pair: tune settings and threshold
 *   train_validation.arff + test_for_train_validation.arff final pair: refit and score the holdout once
 *
 * Each pair shares nominal domains learned from its own training file, so the two pairs are
 * checked separately. test.arff holds the same holdout rows as test_for_train_validation.arff
 * in the development schema; it is not needed here.
 */
public final class DataLoader {

    /** The four files of one data folder and the index of the positive class value. */
    public record Data(String name, Instances train, Instances val, Instances trainVal, Instances test, int pos) {
        /** The positive class label, e.g. "readmitted". */
        public String positive() { return train.classAttribute().value(pos); }
    }

    private DataLoader() {}

    /**
     * Load the four ARFF files from `dir`, use the last column as the class, and fail early if a
     * pair does not share one header, train_validation is not train + validation, or the positive
     * label is missing.
     */
    public static Data load(Path dir, String positive) throws Exception {
        Instances train = read(dir, "train");
        Instances val = read(dir, "validation");
        Instances trainVal = read(dir, "train_validation");
        Instances test = read(dir, "test_for_train_validation");
        if (!train.equalHeaders(val)) throw new IllegalArgumentException("validation vs train: " + train.equalHeadersMsg(val));
        if (!trainVal.equalHeaders(test)) throw new IllegalArgumentException("test vs train_validation: " + trainVal.equalHeadersMsg(test));
        if (trainVal.numInstances() != train.numInstances() + val.numInstances()) {
            throw new IllegalArgumentException("train_validation has " + trainVal.numInstances() + " rows, expected train + validation = "
                    + (train.numInstances() + val.numInstances()));
        }
        int pos = train.classAttribute().indexOfValue(positive);
        if (pos < 0) throw new IllegalArgumentException("class has no value '" + positive + "': " + train.classAttribute());
        if (trainVal.classAttribute().indexOfValue(positive) != pos) {
            throw new IllegalArgumentException("'" + positive + "' has a different class index in train_validation");
        }
        return new Data(dir.toAbsolutePath().normalize().getFileName().toString(), train, val, trainVal, test, pos);
    }

    /** Read one file, e.g. data/weka/train.arff, with the last column as the class. */
    private static Instances read(Path dir, String split) throws Exception {
        Path file = dir.resolve(split + ".arff");
        if (!Files.exists(file)) {
            throw new IOException("missing " + file + " (run notebooks/02_split_export.ipynb to create it)");
        }
        Instances data = DataSource.read(file.toString());
        data.setClassIndex(data.numAttributes() - 1);
        return data;
    }
}
