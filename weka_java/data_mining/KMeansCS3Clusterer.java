package weka_java.data_mining;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import weka.clusterers.SimpleKMeans;
import weka.core.Attribute;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.Standardize;

/**
 * Clustering C2: K-Means with k = 5 on feature space CS3 (8 numeric utilization/complexity
 * features).
 *
 * <p>Self-contained: takes plain {@link Instances} only, so it does not depend on how the data
 * was loaded. Wire it to {@code model.DataLoader.Data} from the caller, for example:
 * <pre>{@code
 *     DataLoader.Data data = DataLoader.load(dir, "readmitted");
 *
 *     // Development: tune on Train, compare on Validation.
 *     KMeansCS3Clusterer devClusterer = new KMeansCS3Clusterer(42);
 *     int[] trainClusters = devClusterer.fitTransform(data.train());
 *     int[] valClusters = devClusterer.transform(data.val());
 *
 *     // Final: refit on Train+Validation, score once on the held-out Test.
 *     KMeansCS3Clusterer finalClusterer = new KMeansCS3Clusterer(42);
 *     int[] trainValClusters = finalClusterer.fitTransform(data.trainVal());
 *     int[] testClusters = finalClusterer.transform(data.test());
 * }</pre>
 * Preprocessing (Standardize) and the cluster centroids are learned only from the split passed to
 * {@link #fitTransform}; {@link #transform} reuses them without refitting.
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

    /** Fits scaling and K-Means on {@code data} and returns its cluster index per row. */
    public int[] fitTransform(Instances data) throws Exception {
        requireNoMissing(data, CS3_FEATURES);

        selectCs3 = new Remove();
        selectCs3.setAttributeIndices(indicesToRemove(data, CS3_FEATURES));
        selectCs3.setInputFormat(data);
        Instances cs3 = Filter.useFilter(data, selectCs3);

        standardize = new Standardize();
        standardize.setInputFormat(cs3);
        Instances scaled = Filter.useFilter(cs3, standardize);

        model = fitBestOfRuns(scaled);
        return assign(scaled);
    }

    /** Assigns clusters to another split using the already-fitted scaler and centroids. */
    public int[] transform(Instances data) throws Exception {
        if (model == null) {
            throw new IllegalStateException("Call fitTransform() before transform().");
        }
        requireNoMissing(data, CS3_FEATURES);
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
        distance.setDontNormalize(true); // distances on z-scores, not Weka's own min-max scaling

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

    /** 1-based indices of every attribute that is not in {@code keep} (for Weka's Remove). */
    private static String indicesToRemove(Instances data, String[] keep) {
        List<String> missing = new ArrayList<>();
        for (String name : keep) {
            if (data.attribute(name) == null) {
                missing.add(name);
            }
        }
        if (!missing.isEmpty()) {
            throw new IllegalArgumentException("Missing CS3 feature(s): " + missing);
        }
        Set<String> keepSet = new HashSet<>(Arrays.asList(keep));
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < data.numAttributes(); i++) {
            if (!keepSet.contains(data.attribute(i).name())) {
                if (builder.length() > 0) {
                    builder.append(',');
                }
                builder.append(i + 1);
            }
        }
        return builder.toString();
    }
}
