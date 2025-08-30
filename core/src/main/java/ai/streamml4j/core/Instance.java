package ai.streamml4j.core;

public class Instance {
    private final double[] features;
    private final double label;

    public Instance(double[] features, double label) {
        this.features = features.clone();
        this.label = label;
    }

    public double[] getFeatures() { return features; }
    public double getLabel() { return label; }
}
