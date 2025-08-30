package ai.streamml4j.algorithms.classification;

import ai.streamml4j.core.Instance;
import ai.streamml4j.core.StreamClassifier;

import java.util.HashMap;
import java.util.Map;

public class NaiveBayesClassifier implements StreamClassifier {
    private final Map<Integer, Map<Integer, Integer>> featureClassCounts = new HashMap<>();

    private final Map<Integer, Integer> classCounts = new HashMap<>();

    private int totalCount = 0;

    private final int vocabSize;     // number of possible feature IDs
    private final double smoothing;  // Laplace smoothing parameter

    public NaiveBayesClassifier(int vocabSize) {
        this(vocabSize, 1.0);
    }

    public NaiveBayesClassifier(int vocabSize, double smoothing) {
        this.vocabSize = vocabSize;
        this.smoothing = smoothing;
    }

    @Override
    public double score(Instance inst) {
        double[] x = inst.getFeatures();

        // compute log-probability for each class
        double[] logProbs = new double[2]; // classes: 0 and 1
        for (int c = 0; c <= 1; c++) {
            int classCount = classCounts.getOrDefault(c, 0);
            if (classCount == 0) {
                logProbs[c] = Double.NEGATIVE_INFINITY;
                continue;
            }

            // prior
            logProbs[c] = Math.log((double) classCount / totalCount);

            // likelihood per feature (assumes features are word counts)
            for (int f = 0; f < x.length; f++) {
                int countInDoc = (int) x[f];
                if (countInDoc <= 0) continue;

                int featureCountGivenClass = featureClassCounts
                        .getOrDefault(c, new HashMap<>())
                        .getOrDefault(f, 0);

                double prob = (featureCountGivenClass + smoothing) /
                        (classCount + smoothing * vocabSize);

                logProbs[c] += countInDoc * Math.log(prob);
            }
        }
        return logProbs[1] - logProbs[0]; // positive score → class 1 more likely
    }

    @Override
    public void learn(Instance inst) {
        int label = (int) inst.getLabel();
        classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);
        totalCount++;

        double[] x = inst.getFeatures();
        featureClassCounts.putIfAbsent(label, new HashMap<>());
        Map<Integer, Integer> counts = featureClassCounts.get(label);

        for (int f = 0; f < x.length; f++) {
            int v = (int) x[f];
            if (v <= 0) continue;
            counts.put(f, counts.getOrDefault(f, 0) + v);
        }
    }
}
