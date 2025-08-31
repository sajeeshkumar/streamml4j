package ai.streamml4j.examples;

import ai.streamml4j.algorithms.classification.PerceptronClassifier;
import ai.streamml4j.core.Instance;
import ai.streamml4j.metrics.Accuracy;

import java.util.Arrays;
import java.util.Set;

public class PerceptronSpamDemo {
    public static void main(String[] args) {
        // Word groups
        Set<String> spamWords = Set.of("buy", "cheap", "viagra", "offer", "free", "money");
        Set<String> hamWords  = Set.of("hello", "project", "meeting", "schedule");

        // Fake dataset: messages and labels
        // label: 1 = spam, 0 = ham
        String[][] messages = {
                {"buy", "cheap", "viagra"}, {"hello", "meeting"}, {"free", "money", "offer"},
                {"project", "meeting"}, {"cheap", "buy"}, {"hello", "project"},
                {"viagra", "cheap"}, {"buy", "offer"}, {"schedule", "meeting"},
                {"free", "viagra", "money"}, {"project", "schedule"}, {"hello"},
                {"buy", "cheap"}, {"meeting", "project"}, {"offer", "free", "money"},
                {"hello", "schedule"}, {"viagra", "free"}, {"buy"}, {"project", "meeting", "schedule"},
                {"cheap", "offer"}
        };
        int[] labels = {
                1,0,1,0,1,0,1,1,0,1,
                0,0,1,0,1,0,1,1,0,1
        };

        // Perceptron with 2 features (spam count, ham count)
        PerceptronClassifier clf = new PerceptronClassifier(2, 0.2);
        Accuracy acc = new Accuracy();

        for (int i = 0; i < messages.length; i++) {
            double[] features = featurize(messages[i], spamWords, hamWords);
            Instance inst = new Instance(features, labels[i]);

            int pred = clf.predict(inst);
            acc.update(labels[i], pred);
            clf.learn(inst);

            System.out.printf("Msg %d: %s | features=%s true=%d pred=%d acc=%.2f%%%n",
                    i + 1, Arrays.toString(messages[i]), Arrays.toString(features),
                    labels[i], pred, 100 * acc.get());
        }

        System.out.printf("Final accuracy after %d messages: %.2f%%%n",
                messages.length, 100 * acc.get());
    }

    /** Count spammy vs hammy words → 2D feature vector */
    private static double[] featurize(String[] words, Set<String> spamWords, Set<String> hamWords) {
        double spamCount = 0, hamCount = 0;
        for (String w : words) {
            if (spamWords.contains(w)) spamCount++;
            if (hamWords.contains(w)) hamCount++;
        }
        return new double[]{spamCount, hamCount};
    }
}
