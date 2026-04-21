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

## 课程作业改造建议（只用基础库）

如果你的课程要求只能使用 `math` 和 Python 基础能力，可以按下面顺序改造：

1. **先替换距离计算**：不要依赖 `numpy/scipy`，改成手写欧氏距离。当前实现使用 `_euclidean_distance`，只依赖 `math.sqrt`。
2. **再替换数据结构辅助库**：例如把 `dataclass` 改为显式 `__init__`，方便老师检查对象初始化流程。
3. **最后替换文件读写封装**：如 `pathlib` 改成 `open(...).readlines()`，确保只使用课程允许的基础 I/O。
4. **每做一步就跑测试**：保证行为不变，再进行下一步。
5. **保留接口不变**：优先保证 `QTClusterer.fit`、`fit_predict`、`load_points` 的调用方式稳定，这样改造后测试更容易通过。

推荐你每次提交只做一类改动（例如“仅替换距离函数”），便于回滚和答辩说明。
