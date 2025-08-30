package ai.streamml4j.core;

public interface StreamClassifier {
    double score(Instance inst);

    default int predict(Instance inst) {
        return score(inst) >= 0.0 ? 1 : 0;
    }

    void learn(Instance inst);
}
