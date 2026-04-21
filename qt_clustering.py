"""QT (Quality Threshold) clustering for multi-dimensional data points."""

import sys
import math


Point = tuple[float, ...]


class QTClusterer:
    """Deterministic QT clustering with a fixed maximum cluster diameter."""

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        if threshold < 0:
            raise ValueError("threshold must be non-negative")

    def fit(self, points: list[Point]) -> list[list[int]]:
        if not points:
            return []

        self._validate_dimensions(points)
        distances = self._distance_matrix(points)
        unassigned = set(range(len(points)))
        clusters: list[list[int]] = []

        while unassigned:
            best_cluster: list[int] | None = None
            best_diameter = math.inf

            for seed in sorted(unassigned):
                candidate, diameter = self._build_cluster(seed, unassigned, distances)
                if best_cluster is None:
                    best_cluster = candidate
                    best_diameter = diameter
                    continue

                if len(candidate) > len(best_cluster):
                    best_cluster = candidate
                    best_diameter = diameter
                elif len(candidate) == len(best_cluster):
                    if diameter < best_diameter:
                        best_cluster = candidate
                        best_diameter = diameter
                    elif math.isclose(diameter, best_diameter) and candidate < best_cluster:
                        best_cluster = candidate
                        best_diameter = diameter

            assert best_cluster is not None
            clusters.append(best_cluster)
            unassigned.difference_update(best_cluster)

        return clusters

    def fit_predict(self, points: list[Point]) -> list[int]:
        clusters = self.fit(points)
        labels = [-1] * len(points)
        for cluster_id, cluster in enumerate(clusters):
            for point_idx in cluster:
                labels[point_idx] = cluster_id
        return labels

    def cluster_points(self, points: list[Point]) -> list[list[Point]]:
        clusters = self.fit(points)
        return [[points[idx] for idx in cluster] for cluster in clusters]

    def _build_cluster(
        self,
        seed: int,
        available: set[int],
        distances: list[list[float]],
    ) -> tuple[list[int], float]:
        cluster = [seed]
        cluster_set = {seed}
        diameter = 0.0

        while True:
            best_next = None
            best_new_diameter = math.inf

            for idx in sorted(available):
                if idx in cluster_set:
                    continue

                new_diameter = diameter
                for existing in cluster:
                    new_diameter = max(new_diameter, distances[idx][existing])
                    if new_diameter > self.threshold:
                        break

                if new_diameter > self.threshold:
                    continue

                if best_next is None or new_diameter < best_new_diameter or (
                    math.isclose(new_diameter, best_new_diameter) and idx < best_next
                ):
                    best_next = idx
                    best_new_diameter = new_diameter

            if best_next is None:
                break

            cluster.append(best_next)
            cluster_set.add(best_next)
            diameter = best_new_diameter

        cluster.sort()
        return cluster, diameter

    @staticmethod
    def _distance_matrix(points: list[Point]) -> list[list[float]]:
        size = len(points)
        distances = [[0.0] * size for _ in range(size)]
        for i in range(size):
            for j in range(i + 1, size):
                dist = _euclidean_distance(points[i], points[j])
                distances[i][j] = dist
                distances[j][i] = dist
        return distances

    @staticmethod
    def _validate_dimensions(points: list[Point]) -> None:
        dims = len(points[0])
        if dims == 0:
            raise ValueError("points must have at least one dimension")
        for idx, point in enumerate(points):
            if len(point) != dims:
                raise ValueError(
                    f"inconsistent point dimensions: point 0 has {dims}, "
                    f"point {idx} has {len(point)}"
                )


def load_points(path: str) -> list[tuple[str, Point]]:
    with open(path, "r", encoding="utf-8") as fp:
        lines = [line.strip() for line in fp.readlines()]
    lines = [line for line in lines if line]
    if not lines:
        return []

    declared_count: int | None = None
    try:
        declared_count = int(lines[0])
        data_lines = lines[1:]
    except ValueError:
        data_lines = lines

    points = [_parse_point(line) for line in data_lines]

    if declared_count is not None and declared_count != len(points):
        raise ValueError(
            f"declared {declared_count} points but found {len(points)} point rows"
        )

    return points


def _parse_point(line: str):
    pieces = line.split()
    label = pieces[0]
    coords = tuple(float(x) for x in pieces[1:])
    return label, coords


def _euclidean_distance(a: Point, b: Point) -> float:
    total = 0.0
    for idx in range(len(a)):
        diff = a[idx] - b[idx]
        total += diff * diff
    return math.sqrt(total)


def _format_point(point) -> str:
    return " ".join(f"{value:g}" for value in point)




def main():
    # 1. parse arguments
    if len(sys.argv) != 3:
        print("Usage: python cluster.py <input_file> <threshold>")
        sys.exit(1)

    input_file = sys.argv[1]
    threshold_arg = sys.argv[2]

    # 2. load data
    data = load_points(input_file)
    
    labels = [label for label, _ in data]
    points = [coords for _, coords in data]

    # 3. compute distance matrix
    temp_clusterer = QTClusterer(0)
    distances = temp_clusterer._distance_matrix(points)

    # 4. process threshold
    if threshold_arg.endswith("%"):
        percent = float(threshold_arg[:-1]) / 100

        max_dist = 0.0
        for i in range(len(distances)):
            for j in range(len(distances)):
                if distances[i][j] > max_dist:
                    max_dist = distances[i][j]

        threshold = percent * max_dist
    else:
        threshold = float(threshold_arg)

    # 5. clustering
    clusterer = QTClusterer(threshold)
    clusters = clusterer.fit(points)

    # 6. output
    for idx, cluster in enumerate(clusters, start=1):
        print(f"Cluster-{idx}")
        for point_idx in cluster:
            label = labels[point_idx]
            point = points[point_idx]
            print(f"{label} {_format_point(point)}")


if __name__ == "__main__":
    main()
