import 'package:ml_knn/ml_knn.dart';

void main() {
  final features = [
    [1.0, 2.0],
    [1.5, 1.8],
    [5.0, 8.0],
    [6.0, 9.0],
  ];
  final labels = ['A', 'A', 'B', 'B'];

  final knn = KNN(k: 3, normalize: true);
  knn.fit(features, labels);

  final prediction = knn.predict([1.2, 1.9]);
  print('Prediction: $prediction');

  final probs = knn.classProbabilities([1.2, 1.9]);
  print('Probabilities: $probs');
}
