package ai.streamml4j.metrics;

public class Accuracy {
    private int correct = 0;
    private int total = 0;

    public void update(double truth, double prediction) {
        if (truth == prediction) {
            correct++;
        }
        total++;
    }

    public double get() {
        return total == 0 ? 0.0 : (double) correct / total;
    }

    public void reset() {
        correct = 0;
        total = 0;
    }
}
