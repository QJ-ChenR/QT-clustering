#!/usr/bin/env python3
"""QT (Quality Threshold) clustering for multi-dimensional data points."""

import sys
import math

Point = tuple[float, ...]


class QTClusterer:
    def __init__(self, threshold: float) -> None:
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self.threshold = threshold
        self.threshold_sq = threshold * threshold

    def fit(
        self,
        points: list[Point],
        distances_sq: list[list[float]] | None = None,
    ) -> list[list[int]]:
        if not points:
            return []

        self._validate_dimensions(points)

        if distances_sq is None:
            distances_sq = self._distance_matrix_sq(points)

        n = len(points)
        neighbours = self._threshold_neighbours(distances_sq, self.threshold_sq)

        unassigned = set(range(n))
        is_available = [True] * n

        in_cluster_mark = [0] * n
        current_mark = 0

        clusters: list[list[int]] = []

        while unassigned:
            best_cluster: list[int] | None = None
            best_diameter_sq = math.inf

            available_sorted = sorted(unassigned)

            for seed in available_sorted:
                current_mark += 1

                candidate, diameter_sq = self._build_cluster(
                    seed=seed,
                    neighbours=neighbours,
                    is_available=is_available,
                    in_cluster_mark=in_cluster_mark,
                    mark=current_mark,
                    distances_sq=distances_sq,
                    current_best_size=0 if best_cluster is None else len(best_cluster),
                    n=n,
                )

                if best_cluster is None:
                    best_cluster = candidate
                    best_diameter_sq = diameter_sq
                elif len(candidate) > len(best_cluster):
                    best_cluster = candidate
                    best_diameter_sq = diameter_sq
                elif len(candidate) == len(best_cluster):
                    if diameter_sq < best_diameter_sq:
                        best_cluster = candidate
                        best_diameter_sq = diameter_sq
                    elif math.isclose(diameter_sq, best_diameter_sq) and candidate < best_cluster:
                        best_cluster = candidate
                        best_diameter_sq = diameter_sq

            assert best_cluster is not None
            clusters.append(best_cluster)

            for idx in best_cluster:
                unassigned.remove(idx)
                is_available[idx] = False

        return clusters

    def _build_cluster(
        self,
        seed: int,
        neighbours: list[list[int]],
        is_available: list[bool],
        in_cluster_mark: list[int],
        mark: int,
        distances_sq: list[list[float]],
        current_best_size: int,
        n: int,
    ) -> tuple[list[int], float]:
        cluster = [seed]
        in_cluster_mark[seed] = mark
        diameter_sq = 0.0

        # Only points that are available AND within threshold of seed can ever join.
        candidates = [
            idx for idx in neighbours[seed]
            if is_available[idx]
        ]

        remaining = len(candidates)

        max_dist_to_cluster = [0.0] * n
        seed_distances = distances_sq[seed]

        for idx in candidates:
            max_dist_to_cluster[idx] = seed_distances[idx]

        threshold_sq = self.threshold_sq
        dist = distances_sq
        in_mark = in_cluster_mark

        while True:
            if len(cluster) + remaining <= current_best_size:
                break

            best_next = -1
            best_new_diameter_sq = math.inf

            for idx in candidates:
                if in_mark[idx] == mark:
                    continue

                candidate_dist_sq = max_dist_to_cluster[idx]

                if candidate_dist_sq > diameter_sq:
                    new_diameter_sq = candidate_dist_sq
                else:
                    new_diameter_sq = diameter_sq

                if new_diameter_sq > threshold_sq:
                    continue

                if (
                    best_next == -1
                    or new_diameter_sq < best_new_diameter_sq
                    or (
                        math.isclose(new_diameter_sq, best_new_diameter_sq)
                        and idx < best_next
                    )
                ):
                    best_next = idx
                    best_new_diameter_sq = new_diameter_sq

            if best_next == -1:
                break

            cluster.append(best_next)
            in_mark[best_next] = mark
            remaining -= 1
            diameter_sq = best_new_diameter_sq

            best_next_distances = dist[best_next]

            for idx in candidates:
                if in_mark[idx] != mark:
                    d_sq = best_next_distances[idx]
                    if d_sq > max_dist_to_cluster[idx]:
                        max_dist_to_cluster[idx] = d_sq

        cluster.sort()
        return cluster, diameter_sq

    @staticmethod
    def _threshold_neighbours(
        distances_sq: list[list[float]],
        threshold_sq: float,
    ) -> list[list[int]]:
        n = len(distances_sq)
        neighbours: list[list[int]] = [[] for _ in range(n)]

        for i in range(n):
            row = distances_sq[i]
            neighbours[i] = [
                j for j in range(n)
                if j != i and row[j] <= threshold_sq
            ]

        return neighbours

    @staticmethod
    def _distance_matrix_sq(points: list[Point]) -> list[list[float]]:
        size = len(points)
        distances_sq = [[0.0] * size for _ in range(size)]

        for i in range(size):
            point_i = points[i]
            for j in range(i + 1, size):
                dist_sq = _squared_euclidean_distance(point_i, points[j])
                distances_sq[i][j] = dist_sq
                distances_sq[j][i] = dist_sq

        return distances_sq

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
        lines = [line.strip() for line in fp if line.strip()]

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


def _parse_point(line: str) -> tuple[str, Point]:
    pieces = line.split()

    if len(pieces) < 2:
        raise ValueError("each row must contain a label and at least one coordinate")

    label = pieces[0]

    try:
        coords = tuple(float(x) for x in pieces[1:])
    except ValueError as exc:
        raise ValueError(f"invalid numeric coordinate in line: {line}") from exc

    return label, coords


def _squared_euclidean_distance(a: Point, b: Point) -> float:
    total = 0.0
    for idx in range(len(a)):
        diff = a[idx] - b[idx]
        total += diff * diff
    return total


def _format_point(point: Point) -> str:
    return " ".join(f"{value:g}" for value in point)


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python qt_clustering.py <input_file> <threshold>")
        sys.exit(1)

    input_file = sys.argv[1]
    threshold_arg = sys.argv[2]

    data = load_points(input_file)

    labels = [label for label, _ in data]
    points = [coords for _, coords in data]

    distances_sq = QTClusterer._distance_matrix_sq(points)

    if threshold_arg.endswith("%"):
        percent = float(threshold_arg[:-1]) / 100
        max_dist_sq = max(max(row) for row in distances_sq)
        threshold = percent * math.sqrt(max_dist_sq)
    else:
        threshold = float(threshold_arg)

    clusterer = QTClusterer(threshold)
    clusters = clusterer.fit(points, distances_sq=distances_sq)

    for idx, cluster in enumerate(clusters, start=1):
        print(f"Cluster-{idx}")
        for point_idx in cluster:
            label = labels[point_idx]
            point = points[point_idx]
            print(f"{label} {_format_point(point)}")


if __name__ == "__main__":
    main()