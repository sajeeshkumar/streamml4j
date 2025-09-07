# StreamML4J

A lightweight **Java library for streaming machine learning**, built for real-time applications where data arrives continuously and models must update online.

This project starts simple with a **production-ready Online Logistic Regression (SGD)** and can grow into a broader toolkit for streaming ML in Java.

---

## 📂 Project Structure

```
streamml4j/
├── core/          # Base abstractions (Instance, OnlineClassifier)
├── algorithms/    # Streaming algorithms (currently: Online Logistic Regression)
├── examples/      # Demo apps using the library
└── pom.xml        # Parent Maven build
```

---

## ⚡ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/sajeeshkumar/streamml4j.git
cd streamml4j
```

### 2. Build with Maven
```bash
mvn clean install
```

### 3. Run the example
```bash
cd examples
mvn exec:java -Dexec.mainClass="ai.streamml4j.examples.LogisticRegressionSpamDemo"
```

Expected output (toy dataset):
```
Msg  1: buy cheap viagra          features=[3.0, 0.0] true=1 pred=0 score=0.50 acc=0.00%
Msg  2: hello project meeting     features=[0.0, 3.0] true=0 pred=0 score=0.42 acc=50.00%
...
Final accuracy: 83.33%
```

---

## 🧩 Core Concepts

- **Instance** → a single training example `(features[], label)`.
- **OnlineClassifier** → interface for any streaming model with:
    - `predict(Instance inst)` → return class (0 or 1).
    - `score(Instance inst)` → probability of positive class.
    - `learn(Instance inst)` → update model with new instance.

---

## 🚀 First Algorithm: Online Logistic Regression (SGD)

- Learns incrementally from one example at a time.
- Uses **stochastic gradient descent (SGD)**.
- Supports **regularization** (`lambda`) to avoid overfitting.
- Suitable for binary classification tasks (e.g. spam vs. ham).

---

## 🌐 Roadmap

- [ ] Add Perceptron (baseline online classifier).
- [ ] Implement Hoeffding Tree (streaming decision trees).
- [ ] Integrate with Kafka/Flink for real-time pipelines.
- [ ] Expand to regression and multi-class classification.

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repo.
2. Create a feature branch.
3. Submit a Pull Request.

All changes must go through PR review and CI (Maven build).

---

## 📜 License

MIT License. See [LICENSE](LICENSE).  
