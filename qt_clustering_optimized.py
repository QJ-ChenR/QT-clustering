"""Optimized QT (Quality Threshold) clustering for multi-dimensional data points."""

import math
import sys


Point = tuple[float, ...]


class QTClusterer:
    """Deterministic QT clustering with a fixed maximum cluster diameter."""

    def __init__(self, threshold: float) -> None:
        # check threshold is non-negative
        self.threshold = threshold
        if threshold < 0:
            raise ValueError("threshold must be non-negative")

    def fit(self, points: list[Point]) -> list[list[int]]:
        if not points:
            return []

        self._validate_dimensions(points) # check all points have the same number of dimensions
        distances = self._distance_matrix(points) # precompute distance matrix for efficiency
        neighbors = self._build_neighbor_lists(distances, self.threshold) # precompute neighbor lists for efficiency

        n_points = len(points)
        active = [True] * n_points
        active_count = n_points
        clusters: list[list[int]] = []

        while active_count > 0:
            best_cluster: list[int] | None = None
            best_diameter = math.inf

            for seed in range(n_points):
                if not active[seed]:
                    continue

                candidate, diameter = self._build_cluster(seed, active, distances, neighbors)
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

            for idx in best_cluster:
                if active[idx]:
                    active[idx] = False
                    active_count -= 1

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
        active: list[bool],
        distances: list[list[float]],
        neighbors: list[list[int]],
    ) -> tuple[list[int], float]:
        cluster = [seed]
        candidates: list[int] = []

        for neighbor in neighbors[seed]:
            # only consider active neighbors that are not the seed itself
            # any point beyond threshold from the seed can never belong to this cluster
            if active[neighbor] and neighbor != seed:
                candidates.append(neighbor)

        diameter_cache: dict[int, float] = {} 
        # diameter_cache[candidate] stores the current maximum distance
        # this reduces the number of distance calculations needed to evaluate candidates
        for candidate in candidates:
            diameter_cache[candidate] = distances[seed][candidate]

        cluster_diameter = 0.0

        while candidates:
            # greedily choose the candidate that minimally increases cluster diameter
            # ties are broken deterministically by smaller point index
            best_candidate: int | None = None
            smallest_diameter = math.inf

            for candidate in candidates:
                candidate_diameter = diameter_cache[candidate]
                if (
                    candidate_diameter < smallest_diameter
                    or (
                        math.isclose(candidate_diameter, smallest_diameter)
                        and (best_candidate is None or candidate < best_candidate)
                    )
                ):
                    smallest_diameter = candidate_diameter
                    best_candidate = candidate

            if best_candidate is None or smallest_diameter > self.threshold:
                break

            cluster.append(best_candidate)
            cluster_diameter = smallest_diameter
            candidates.remove(best_candidate)

            for candidate in candidates:
                updated = distances[best_candidate][candidate]
                if updated > diameter_cache[candidate]:
                    diameter_cache[candidate] = updated

        cluster.sort()
        return cluster, cluster_diameter

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
    def _build_neighbor_lists(
        distances: list[list[float]],
        threshold: float,
    ) -> list[list[int]]:
    # Neighbor lists restrict search space to threshold-compatible points only
        size = len(distances)
        neighbors: list[list[int]] = []
        for i in range(size):
            row_neighbors: list[int] = []
            for j in range(size):
                if i != j and distances[i][j] <= threshold:
                    row_neighbors.append(j)
            neighbors.append(row_neighbors)
        return neighbors

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
    if len(sys.argv) != 3:
        print("Usage: python cluster.py <input_file> <threshold>")
        sys.exit(1)

    # threshold can be a float or a percentage string like "20%"
    input_file = sys.argv[1]
    threshold_arg = sys.argv[2]

    # load points and separate labels from coordinates
    data = load_points(input_file)
    labels = [label for label, _ in data]
    points = [coords for _, coords in data]

    temp_clusterer = QTClusterer(0)
    distances = temp_clusterer._distance_matrix(points)

    if threshold_arg.endswith("%"): # interpret percentage threshold relative to maximum distance in the dataset
        percent = float(threshold_arg[:-1]) / 100
        max_dist = 0.0
        for i in range(len(distances)):
            for j in range(len(distances)):
                if distances[i][j] > max_dist:
                    max_dist = distances[i][j]
        threshold = percent * max_dist
    else:
        threshold = float(threshold_arg)

    clusterer = QTClusterer(threshold) # initialize clusterer with the specified threshold
    clusters = clusterer.fit(points) # compute clusters based on the input points and threshold

    # print clusters with labels and coordinates
    for idx, cluster in enumerate(clusters, start=1):
        print(f"Cluster-{idx}")
        for point_idx in cluster:
            label = labels[point_idx]
            point = points[point_idx]
            print(f"{label} {_format_point(point)}")


if __name__ == "__main__":
    main()