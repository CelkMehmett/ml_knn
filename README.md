# 📦 ml_knn

A production-ready **K-Nearest Neighbors (KNN)** classifier implemented in pure Dart.  
Supports **Euclidean**, **Manhattan**, and **Cosine** distances, with optional **weighting** and **normalization**.  
Perfect for lightweight ML tasks on **mobile**, **web**, or **server-side Dart**.

---

## ✨ Features

- 🧠 Distance metrics: Euclidean, Manhattan, Cosine
- ⚖️ Optional weighted voting
- 📊 Built-in normalization support
- 🔁 Batch predictions
- 💾 Model serialization/deserialization
- 🧪 Full test coverage

---

## 🚀 Quick Start

```dart
import 'package:ml_knn/ml_knn.dart';

void main() {
  final features = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
  ];
  final labels = ['A', 'B', 'A'];

  final knn = KNN(k: 3, normalize: true);
  knn.fit(features, labels);

  final prediction = knn.predict([1.2, 0.1]);
  print('Prediction: $prediction');

  final probs = knn.classProbabilities([1.2, 0.1]);
  print('Probabilities: $probs');
}
