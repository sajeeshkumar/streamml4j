package ai.streamml4j.core;

public interface OnlineClassifier {
    int predict(Instance inst);
    double score(Instance inst);
    void learn(Instance inst);
}
