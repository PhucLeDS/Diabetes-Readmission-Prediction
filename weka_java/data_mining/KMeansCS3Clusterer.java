package diabetes.datamining;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import weka.clusterers.SimpleKMeans;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Standardize;

/**
 * Clustering config C2: K-Means with k = 5 on feature space CS3.
 *
 * <p>Mirrors {@code PatientClusterer(config_id="C2", feature_space="CS3")} in
 * clustering.py:
 * <ul>
 *   <li>Z-score scaling is learned from Train only (Weka Standardize).</li>
 *   <li>K-Means uses k-means++ initialisation and {@value #N_INIT} restarts with
 *       seeds {@code randomState + run}. The run with the smallest squared error
 *       is kept, similar to scikit-learn's {@code n_init}.</li>
 *   <li>Weka's EuclideanDistance is set to {@code dontNormalize}, so distances are
 *       computed on the z-scores as in scikit-learn. Without this, SimpleKMeans
 *       would min-max normalise the data a second time.</li>
 * </ul>
 * Labels from Weka and scikit-learn can differ by permutation and by tie or
 * convergence details. Compare with the parity test, not by exact equality.
 */
public final class KMeansCS3Clusterer {

    public static final String[] CS3_FEATURES = {
            "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
            "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"};

    public static final int N_CLUSTERS = 5;
    public static final int N_INIT = 10;
    public static final int MAX_ITERATIONS = 300;

    private final int randomState;
    private Remove selectCs3;
    private Standardize standardize;
    private SimpleKMeans model;

    public KMeansCS3Clusterer(int randomState) {
        this.randomState = randomState;
    }

    /** Fits scaling and K-Means on Train and returns the Train cluster index of each row. */
    public int[] fitTransform(Instances train) throws Exception {
        D0Loader.requireNoMissing(train, CS3_FEATURES);

        selectCs3 = new Remove();
        selectCs3.setAttributeIndices(indicesToRemove(train, CS3_FEATURES));
        selectCs3.setInputFormat(train);
        Instances cs3 = Filter.useFilter(train, selectCs3);

        standardize = new Standardize();
        standardize.setInputFormat(cs3);
        Instances scaled = Filter.useFilter(cs3, standardize);

        model = fitBestOfRuns(scaled);
        return assign(scaled);
    }

    /** Assigns clusters to Validation or Test rows with the Train-fitted scaler and centroids. */
    public int[] transform(Instances data) throws Exception {
        if (model == null) {
            throw new IllegalStateException("Call fitTransform() before transform().");
        }
        D0Loader.requireNoMissing(data, CS3_FEATURES);
        Instances cs3 = Filter.useFilter(data, selectCs3);
        Instances scaled = Filter.useFilter(cs3, standardize);
        return assign(scaled);
    }

    private SimpleKMeans fitBestOfRuns(Instances scaled) throws Exception {
        SimpleKMeans best = null;
        double bestError = Double.POSITIVE_INFINITY;
        for (int run = 0; run < N_INIT; run++) {
            SimpleKMeans candidate = newKMeans(randomState + run);
            candidate.buildClusterer(scaled);
            double error = candidate.getSquaredError();
            if (best == null || error < bestError) {
                best = candidate;
                bestError = error;
            }
        }
        return best;
    }

    private static SimpleKMeans newKMeans(int seed) throws Exception {
        EuclideanDistance distance = new EuclideanDistance();
        distance.setDontNormalize(true);

        SimpleKMeans kmeans = new SimpleKMeans();
        kmeans.setNumClusters(N_CLUSTERS);
        kmeans.setSeed(seed);
        kmeans.setMaxIterations(MAX_ITERATIONS);
        kmeans.setInitializationMethod(
                new SelectedTag(SimpleKMeans.KMEANS_PLUS_PLUS, SimpleKMeans.TAGS_SELECTION));
        kmeans.setDistanceFunction(distance);
        kmeans.setPreserveInstancesOrder(true);
        return kmeans;
    }

    private int[] assign(Instances scaled) throws Exception {
        int[] labels = new int[scaled.numInstances()];
        for (int i = 0; i < labels.length; i++) {
            labels[i] = model.clusterInstance(scaled.instance(i));
        }
        return labels;
    }

    /** 1-based indices of every attribute that is not in {@code keep}. */
    static String indicesToRemove(Instances data, String[] keep) {
        Set<String> keepSet = new HashSet<>(Arrays.asList(keep));
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < data.numAttributes(); i++) {
            String name = data.attribute(i).name();
            if (!keepSet.contains(name)) {
                if (builder.length() > 0) {
                    builder.append(',');
                }
                builder.append(i + 1);
            }
        }
        List<String> found = new ArrayList<>();
        for (String name : keep) {
            if (data.attribute(name) == null) {
                found.add(name);
            }
        }
        if (!found.isEmpty()) {
            throw new IllegalArgumentException("Missing CS3 feature(s): " + found);
        }
        return builder.toString();
    }
}
