# QT-clustering

Deterministic Quality Threshold (QT) clustering for multi-dimensional vectors.

## Usage

```bash
python qt_clustering.py <input_file> <threshold>
```

- `threshold`: maximum allowed cluster diameter (Euclidean distance).
- Output lists clusters and the point indices assigned to each cluster.

## Input format

`<input_file>` supports either:

1. First line is the number of points, followed by one point per line, or
2. One point per line with no count header.

Each point line may be whitespace-separated or comma-separated numeric values.

## Programmatic use

```python
from qt_clustering import QTClusterer, load_points

points = load_points("data.txt")
clusters = QTClusterer(threshold=1.0).fit(points)
```

