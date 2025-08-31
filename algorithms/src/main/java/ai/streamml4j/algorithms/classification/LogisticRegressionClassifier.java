package ai.streamml4j.algorithms.classification;

import ai.streamml4j.core.Instance;
import ai.streamml4j.core.StreamClassifier;

/**
 * Online Logistic Regression using SGD updates.
 */
public class LogisticRegressionClassifier implements StreamClassifier {
    private final double[] weights;
    private double bias;
    private final double lr; // learning rate

    public LogisticRegressionClassifier(int nFeatures, double lr) {
        this.weights = new double[nFeatures];
        this.bias = 0.0;
        this.lr = lr;
    }

    @Override
    public double score(Instance instance) {
        double z = bias;
        double[] x = instance.getFeatures();
        for (int i = 0; i < weights.length; i++) {
            z += weights[i] * x[i];
        }
        return 1.0 / (1.0 + Math.exp(-z));
    }

    @Override
    public int predict(Instance instance) {
        return score(instance) >= 0.5 ? 1 : 0;
    }

    @Override
    public void learn(Instance instance) {
        double[] x = instance.getFeatures();
        double y = instance.getLabel();

        double yHat = score(instance);
        double error = y - yHat; // gradient part

        // SGD weight update
        for (int i = 0; i < weights.length; i++) {
            weights[i] += lr * error * x[i];
        }
        bias += lr * error;
    }
}
