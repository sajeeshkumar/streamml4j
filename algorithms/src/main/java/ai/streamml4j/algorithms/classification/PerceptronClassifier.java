package ai.streamml4j.algorithms.classification;

import ai.streamml4j.core.Instance;
import ai.streamml4j.core.StreamClassifier;

public class PerceptronClassifier implements StreamClassifier {

    private final double[] weights;
    private double bias = 0.0;
    private final double learningRate;

    public PerceptronClassifier(int nFeatures) {
        this(nFeatures, 0.1);
    }

    public PerceptronClassifier(int nFeatures, double learningRate) {
        if (nFeatures <= 0) throw new IllegalArgumentException("nFeatures must be > 0");
        this.weights = new double[nFeatures];
        this.learningRate = learningRate;
    }

    @Override
    public double score(Instance inst) {
        double[] x = inst.getFeatures();
        if (x.length != weights.length) {
            throw new IllegalArgumentException("feature length mismatch: expected " + weights.length + " but got " + x.length);
        }
        double s = bias;
        for (int i = 0; i < weights.length; i++) s += weights[i] * x[i];
        return s;
    }

    @Override
    public void learn(Instance inst) {
        double label = inst.getLabel();
        int y = label == 1.0 ? 1 : -1; // map 0.0 -> -1, 1.0 -> +1
        double s = score(inst);
        if (y * s <= 0.0) { // misclassified (or exactly on boundary)
            double[] x = inst.getFeatures();
            for (int i = 0; i < weights.length; i++) {
                weights[i] += learningRate * y * x[i];
            }
            bias += learningRate * y;
        }
    }
}
