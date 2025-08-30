package ai.streamml4j.examples;

import ai.streamml4j.algorithms.classification.NaiveBayesClassifier;
import ai.streamml4j.core.Instance;
import ai.streamml4j.metrics.Accuracy;

import java.util.HashMap;
import java.util.Map;

/**
 * Streaming Naive Bayes spam/ham demo with toy text messages.
 */
public class NaiveBayesSpamDemo {
    public static void main(String[] args) {
        // Vocabulary of words (toy, small)
        String[] vocab = {"buy", "cheap", "viagra", "hello", "meeting", "project", "offer", "free", "money", "schedule"};
        Map<String, Integer> wordToIndex = new HashMap<>();
        for (int i = 0; i < vocab.length; i++) {
            wordToIndex.put(vocab[i], i);
        }

        // Larger fake dataset (50+ messages)
        // Label: 1 = spam, 0 = ham
        String[][] messages = {
                {"buy", "cheap", "viagra"}, {"hello", "meeting"}, {"free", "money", "offer"},
                {"project", "meeting"}, {"cheap", "buy"}, {"hello", "project"},
                {"viagra", "cheap"}, {"buy", "offer"}, {"schedule", "meeting"},
                {"free", "viagra", "money"}, {"project", "schedule"}, {"hello"},
                {"buy", "cheap"}, {"meeting", "project"}, {"offer", "free", "money"},
                {"hello", "schedule"}, {"viagra", "free"}, {"buy"}, {"project", "meeting", "schedule"},
                {"cheap", "offer"}, {"hello", "project", "meeting"}, {"money", "buy"}, {"schedule"},
                {"buy", "cheap", "offer"}, {"meeting"}, {"free", "money"}, {"project"},
                {"viagra", "buy"}, {"schedule", "project"}, {"cheap", "free"},
                {"hello", "meeting"}, {"buy", "money"}, {"offer"}, {"hello", "project"},
                {"viagra"}, {"free", "buy"}, {"schedule"}, {"project", "meeting"},
                {"cheap", "money"}, {"hello"}, {"buy", "offer", "free"}, {"meeting"},
                {"cheap"}, {"hello", "schedule"}, {"viagra", "cheap"}, {"free", "offer"},
                {"buy", "money"}, {"meeting", "project"}, {"schedule", "hello"}, {"offer", "cheap"}
        };

        int[] labels = {
                1,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,1,1,0,1,
                0,1,0,1,0,1,0,1,0,1,0,1,1,1,0,1,0,1,0,1,
                0,1,1,0,1,0,1,0,1,0,0,1
        };

        // Create Naive Bayes
        NaiveBayesClassifier nb = new NaiveBayesClassifier(vocab.length, 1.0);
        Accuracy acc = new Accuracy();

        // Stream through
        for (int i = 0; i < messages.length; i++) {
            double[] features = featurize(messages[i], wordToIndex, vocab.length);
            Instance inst = new Instance(features, labels[i]);

            int pred = nb.predict(inst);
            acc.update(labels[i], pred);
            nb.learn(inst);

            if ((i + 1) % 10 == 0 || i == messages.length - 1) {
                System.out.printf("After %d messages: accuracy = %.2f%%%n",
                        i + 1, 100 * acc.get());
            }
        }

        System.out.printf("Final accuracy after %d messages: %.2f%%%n",
                messages.length, 100 * acc.get());
    }

    /** Convert words into bag-of-words vector */
    private static double[] featurize(String[] words, Map<String, Integer> wordToIndex, int vocabSize) {
        double[] vec = new double[vocabSize];
        for (String w : words) {
            Integer idx = wordToIndex.get(w);
            if (idx != null) vec[idx] += 1.0;
        }
        return vec;
    }
}
