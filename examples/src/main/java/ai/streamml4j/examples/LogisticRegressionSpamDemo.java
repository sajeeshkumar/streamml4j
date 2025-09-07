package ai.streamml4j.examples;

import ai.streamml4j.algorithms.classification.OnlineLogisticRegression;
import ai.streamml4j.core.Instance;

import java.util.Arrays;

public class LogisticRegressionSpamDemo {
    public static void main(String[] args) {
        String[][] messages = {
                {"buy", "cheap", "viagra"}, // spam
                {"hello", "project", "meeting"}, // ham
                {"cheap", "buy"}, // spam
                {"project", "meeting"}, // ham
                {"free", "money"}, // spam
                {"team", "update"}, // ham
        };
        int[] labels = {1, 0, 1, 0, 1, 0};

        OnlineLogisticRegression clf = new OnlineLogisticRegression(2, 0.1, 0.01);

        int correct = 0;
        for (int i = 0; i < messages.length; i++) {
            double[] features = featurize(messages[i]);
            Instance inst = new Instance(features, labels[i]);

            int pred = clf.predict(inst);
            double score = clf.score(inst);

            if (pred == (int) inst.label()) correct++;
            double acc = 100.0 * correct / (i + 1);

            clf.learn(inst);

            System.out.printf(
                    "Msg %2d: %-25s features=%s true=%d pred=%d score=%.2f acc=%.2f%%%n",
                    i + 1, String.join(" ", messages[i]),
                    Arrays.toString(features),
                    (int) inst.label(), pred, score, acc
            );
        }

        System.out.printf("Final accuracy: %.2f%%%n", 100.0 * correct / messages.length);
    }

    private static double[] featurize(String[] words) {
        int spamCount = 0, hamCount = 0;
        for (String w : words) {
            if ("buy cheap viagra free money".contains(w)) spamCount++;
            else hamCount++;
        }
        return new double[]{spamCount, hamCount};
    }
}
