import os
import tempfile
import unittest

from qt_clustering import QTClusterer, load_points


class TestQTClustering(unittest.TestCase):
    def test_assigns_each_point_once(self):
        points = [
            (0.0, 0.0),
            (0.1, 0.0),
            (5.0, 5.0),
            (5.1, 5.0),
            (10.0, 10.0),
        ]

        clusterer = QTClusterer(threshold=0.5)
        clusters = clusterer.fit(points)

        self.assertEqual(sorted(clusters), [[0, 1], [2, 3], [4]])
        assigned = sorted(idx for cluster in clusters for idx in cluster)
        self.assertEqual(assigned, list(range(len(points))))

    def test_deterministic_repeated_runs(self):
        points = [
            (0.0, 0.0),
            (0.0, 1.0),
            (1.0, 0.0),
            (10.0, 10.0),
        ]

        clusterer = QTClusterer(threshold=1.1)
        first = clusterer.fit(points)
        second = clusterer.fit(points)

        self.assertEqual(first, second)

    def test_load_points_supports_declared_count(self):
        with tempfile.NamedTemporaryFile("w", delete=False) as fp:
            fp.write("3\n1 2\n3 4\n5 6\n")
            temp_path = fp.name

        try:
            points = load_points(temp_path)
        finally:
            os.unlink(temp_path)

        self.assertEqual(points, [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)])


if __name__ == "__main__":
    unittest.main()
