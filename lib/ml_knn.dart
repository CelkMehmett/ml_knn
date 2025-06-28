library ml_knn;

import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

/// Supported distance metrics
enum DistanceMetric { euclidean, manhattan, cosine }

class KNN {
  late List<List<double>> _features;
  late List<dynamic> _labels;

  final int k;
  final DistanceMetric distanceMetric;
  final bool weighted;
  final bool normalize;

  List<double> _means = [];
  List<double> _stds = [];

  KNN({
    required this.k,
    this.distanceMetric = DistanceMetric.euclidean,
    this.weighted = false,
    this.normalize = false,
  });

  void fit(List<List<double>> features, List<dynamic> labels) {
    _features = features.map((f) => f.map((e) => e.toDouble()).toList()).toList();
    _labels = labels;

    if (normalize) {
      _computeNormalization();
      _features = _features.map(_normalizeSample).toList();
    }
  }

  dynamic predict(List<double> input) {
    final probs = classProbabilities(input);
    return probs.entries.reduce((a, b) => a.value > b.value ? a : b).key;
  }

  List<dynamic> batchPredict(List<List<double>> inputs) {
    return inputs.map(predict).toList();
  }

  Map<dynamic, double> classProbabilities(List<double> input) {
    final inputVector = normalize ? _normalizeSample(input) : input;

    final neighbors = _features.asMap().entries.map((e) {
      final distance = _calculateDistance(e.value, inputVector);
      return {'label': _labels[e.key], 'distance': distance};
    }).toList()
      ..sort((a, b) => a['distance'].compareTo(b['distance']));

    final topK = neighbors.take(k);
    final labelWeights = <dynamic, double>{};

    for (var neighbor in topK) {
      final label = neighbor['label'];
      final distance = neighbor['distance'] as double;
      final weight = weighted ? 1.0 / (distance + 1e-8) : 1.0;
      labelWeights[label] = (labelWeights[label] ?? 0.0) + weight;
    }

    final total = labelWeights.values.reduce((a, b) => a + b);
    return labelWeights.map((label, weight) => MapEntry(label, weight / total));
  }

  List<MapEntry<dynamic, double>> predictTopN(List<double> input, int n) {
    final probs = classProbabilities(input);
    final sorted = probs.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));
    return sorted.take(n).toList();
  }

  String toJsonString() {
    return jsonEncode({
      'features': _features,
      'labels': _labels,
      'k': k,
      'distanceMetric': distanceMetric.name,
      'weighted': weighted,
      'normalize': normalize,
      'means': normalize ? _means : null,
      'stds': normalize ? _stds : null,
    });
  }

  void fromJsonString(String jsonStr) {
    final data = jsonDecode(jsonStr);
    _features = List<List<dynamic>>.from(data['features'])
        .map((e) => List<double>.from(e.map((v) => v.toDouble())))
        .toList();
    _labels = data['labels'];
    
    if (data['means'] != null) {
      _means = List<double>.from(data['means']);
    }
    if (data['stds'] != null) {
      _stds = List<double>.from(data['stds']);
    }
  }

  Future<void> saveToFile(String path) async {
    final file = File(path);
    await file.writeAsString(toJsonString());
  }

  Future<void> loadFromFile(String path) async {
    final file = File(path);
    fromJsonString(await file.readAsString());
  }

  void _computeNormalization() {
    if (_features.isEmpty) return;
    
    final n = _features.length;
    final d = _features[0].length;
    
    _means = List.generate(d, (i) => 
        _features.map((x) => x[i]).reduce((a, b) => a + b) / n);
    
    _stds = List.generate(d, (i) {
      final mean = _means[i];
      final variance = _features
          .map((x) => math.pow(x[i] - mean, 2).toDouble())
          .reduce((a, b) => a + b) / n;
      return math.sqrt(variance);
    });
  }

  List<double> _normalizeSample(List<double> input) {
    return List.generate(input.length, (i) => 
        (input[i] - _means[i]) / (_stds[i] + 1e-8));
  }

  double _calculateDistance(List<double> a, List<double> b) {
    switch (distanceMetric) {
      case DistanceMetric.euclidean:
        final sumSquaredDiffs = a.asMap().entries
            .map((e) => math.pow(e.value - b[e.key], 2).toDouble())
            .reduce((x, y) => x + y);
        return math.sqrt(sumSquaredDiffs);
      case DistanceMetric.manhattan:
        return a.asMap().entries
            .map((e) => (e.value - b[e.key]).abs())
            .reduce((x, y) => x + y);
      case DistanceMetric.cosine:
        final dotAB = _dot(a, b);
        final normA = math.sqrt(_dot(a, a));
        final normB = math.sqrt(_dot(b, b));
        return 1 - (dotAB / (normA * normB + 1e-8));
    }
  }

  double _dot(List<double> a, List<double> b) =>
      a.asMap().entries.map((e) => e.value * b[e.key]).reduce((x, y) => x + y);
}