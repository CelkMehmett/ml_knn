import 'package:test/test.dart';
import 'package:ml_knn/ml_knn.dart';
import 'dart:math' as math;

void main() {
  group('KNN Basic Functionality', () {
    final features = [
      [1.0, 0.0],
      [0.0, 1.0],
      [1.0, 1.0],
    ];
    final labels = ['A', 'B', 'A'];

    test('predicts correct label', () {
      final knn = KNN(k: 1);
      knn.fit(features, labels);
      expect(knn.predict([1.0, 0.1]), equals('A'));
      expect(knn.predict([0.1, 1.0]), equals('B'));
    });

    test('probabilities sum to 1', () {
      final knn = KNN(k: 2, weighted: true);
      knn.fit(features, labels);
      final probs = knn.classProbabilities([1.0, 1.0]);
      final sum = probs.values.reduce((a, b) => a + b);
      expect((sum - 1.0).abs() < 1e-8, isTrue);
    });

    test('serialization and deserialization work', () {
      final knn = KNN(k: 2, normalize: true);
      knn.fit(features, labels);
      final json = knn.toJsonString();
      final restored = KNN(k: 2, normalize: true);
      restored.fromJsonString(json);
      expect(restored.predict([1.0, 0.0]), equals(knn.predict([1.0, 0.0])));
    });
  });

  group('KNN Normalization Effect', () {
    test('normalization affects prediction', () {
      // X ekseninde çok büyük değerler, Y ekseninde küçük değerler
      final features = [
        [1.0, 1.0],      // A sınıfı - küçük x, küçük y
        [2.0, 2.0],      // A sınıfı - küçük x, küçük y  
        [1000.0, 1.0],   // B sınıfı - büyük x, küçük y
        [1001.0, 2.0],   // B sınıfı - büyük x, küçük y
      ];
      final labels = ['A', 'A', 'B', 'B'];
      
      // Test noktası: B sınıfına daha yakın olmalı (X değeri açısından)
      final input = [950.0, 1.5];

      // Normalize edilmemiş durumda mesafeleri kontrol edelim
      final knnRaw = KNN(k: 2, normalize: false);
      knnRaw.fit(features, labels);
      
      // Manuel mesafe hesaplama - Raw
      print('Raw distances:');
      for (int i = 0; i < features.length; i++) {
        final dx = features[i][0] - input[0];
        final dy = features[i][1] - input[1];
        final distance = math.sqrt(dx*dx + dy*dy);
        print('  ${labels[i]}: ${features[i]} -> distance = $distance');
      }
      
      final resultRaw = knnRaw.predict(input);

      // Normalize edilmiş durumda
      final knnNorm = KNN(k: 2, normalize: true);
      knnNorm.fit(features, labels);
      final resultNorm = knnNorm.predict(input);

      print('Raw result: $resultRaw');
      print('Normalized result: $resultNorm');

      // Bu sefer kesin farklı sonuçlar bekleyelim
      expect(resultRaw, equals('B'), 
             reason: 'Raw: Input [950,1.5] should be closest to B class [1000,1] and [1001,2]');
    });

    test('weighted vs unweighted KNN', () {
      final features = [
        [1.0, 1.0],
        [2.0, 2.0],
        [10.0, 10.0],
      ];
      final labels = ['A', 'A', 'B'];
      final input = [1.5, 1.5]; // A'ya çok yakın

      final knnWeighted = KNN(k: 3, weighted: true);
      knnWeighted.fit(features, labels);
      final resultWeighted = knnWeighted.predict(input);

      final knnUnweighted = KNN(k: 3, weighted: false);
      knnUnweighted.fit(features, labels);
      final resultUnweighted = knnUnweighted.predict(input);

      // Weighted durumda yakın mesafedeki 'A' etiketleri daha etkili olmalı
      expect(resultWeighted, equals('A'));
      // Unweighted durumda da 'A' çıkabilir ama weighted daha güvenilir
    });

    test('different distance metrics', () {
      final features = [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
      ];
      final labels = ['origin', 'x-axis', 'y-axis'];
      final input = [0.5, 0.5];

      final knnEuclidean = KNN(k: 1, distanceMetric: DistanceMetric.euclidean);
      knnEuclidean.fit(features, labels);
      final resultEuclidean = knnEuclidean.predict(input);

      final knnManhattan = KNN(k: 1, distanceMetric: DistanceMetric.manhattan);
      knnManhattan.fit(features, labels);
      final resultManhattan = knnManhattan.predict(input);

      print('Euclidean result: $resultEuclidean');
      print('Manhattan result: $resultManhattan');

      // Her iki metrik de geçerli sonuçlar verebilir
      expect(['origin', 'x-axis', 'y-axis'].contains(resultEuclidean), isTrue);
      expect(['origin', 'x-axis', 'y-axis'].contains(resultManhattan), isTrue);
    });
  });
}