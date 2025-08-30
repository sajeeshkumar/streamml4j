package ai.streamml4j.metrics;

public class Accuracy {
    private int correct = 0;
    private int total = 0;

    public void update(int yTrue, int yPred) {
        if (yTrue == yPred) correct++;
        total++;
    }

    public double get() {
        return total == 0 ? 0.0 : (double) correct / total;
    }
}
