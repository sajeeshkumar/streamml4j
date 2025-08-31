# streamml4j

🚀 **streamml4j** is an experimental Java library for **online / streaming machine learning**  
(inspired by Python’s [river](https://github.com/online-ml/river)).

## Features (so far)
- ✅ `NaiveBayesClassifier` (incremental multinomial NB)
- ✅ `PerceptronClassifier` (Online Linear Classifier)

- ✅ Simple metrics (`Accuracy`)
- ✅ Example demos (`NaiveBayesSpamDemo`,`PerceptronSpamDemo`)

## Roadmap
- [ ] Logistic Regression (online SGD)
- [ ] Hoeffding Trees (VFDT)
- [ ] Drift detection (ADWIN, DDM)
- [ ] Feature hashing / sketches
- [ ] Benchmarks vs Python’s river

## Build & Run
```bash
mvn clean install
mvn -pl examples exec:java -Dexec.mainClass="ai.streamml4j.examples.NaiveBayesSpamDemo"
