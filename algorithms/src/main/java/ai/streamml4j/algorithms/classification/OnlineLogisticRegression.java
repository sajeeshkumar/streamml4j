package ai.streamml4j.algorithms.classification;

import ai.streamml4j.core.Instance;
import ai.streamml4j.core.OnlineClassifier;

import java.util.Arrays;

public class OnlineLogisticRegression implements OnlineClassifier {

    private final double[] weights;
    private final double learningRate;
    private final double lambda;

    public OnlineLogisticRegression(int nFeatures, double learningRate, double lambda) {
        this.weights = new double[nFeatures];
        this.learningRate = learningRate;
        this.lambda = lambda;
    }

    @Override
    public int predict(Instance inst) {
        return score(inst) >= 0.5 ? 1 : 0;
    }

    @Override
    public double score(Instance inst) {
        double z = 0.0;
        for (int i = 0; i < inst.features().length; i++) {
            z += weights[i] * inst.features()[i];
        }
        return 1.0 / (1.0 + Math.exp(-z));
    }

    @Override
    public void learn(Instance inst) {
        double y = inst.label();
        double p = score(inst);

        for (int i = 0; i < inst.features().length; i++) {
            double grad = (y - p) * inst.features()[i] - lambda * weights[i];
            weights[i] += learningRate * grad;
        }
    }

    @Override
    public String toString() {
        return "OnlineLogisticRegression{" + Arrays.toString(weights) + "}";
    }
}
