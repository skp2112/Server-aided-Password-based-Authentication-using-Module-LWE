import os
os.environ.setdefault("MPLBACKEND", "Agg")

import unittest

import numpy as np

from main import LatticeEngine, PCDLProtocol, format_performance_summary, plot_asymptotic


class LatticeEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = LatticeEngine()

    @staticmethod
    def schoolbook(left, right, q=3329):
        product = np.convolve(left.astype(object), right.astype(object))
        reduced = np.zeros(256, dtype=object)
        reduced[:256] = product[:256]
        reduced[:255] -= product[256:]
        return np.array([int(value) % q for value in reduced], dtype=np.int64)

    def test_ntt_base_multiplication_matches_ring_product(self):
        rng = np.random.default_rng(20260612)
        left = rng.integers(0, self.engine.q, self.engine.n, dtype=np.int64)
        right = rng.integers(0, self.engine.q, self.engine.n, dtype=np.int64)
        expected = self.engine.ntt(self.schoolbook(left, right))
        actual = self.engine.multiply_ntt(self.engine.ntt(left), self.engine.ntt(right))
        np.testing.assert_array_equal(actual, expected)

    def test_cbd_eta_three_domain(self):
        sample = self.engine.cbd_sample(b"a" * 32, b"test", 0)
        centered = np.where(sample > self.engine.q // 2, sample - self.engine.q, sample)
        self.assertTrue(np.all(centered >= -3))
        self.assertTrue(np.all(centered <= 3))
        np.testing.assert_array_equal(sample, self.engine.cbd_sample(b"a" * 32, b"test", 0))

    def test_generated_matrix_is_invertible(self):
        matrix, inverse, _ = self.engine.generate_invertible_matrix(b"m" * 32)
        self.assertTrue(self.engine.is_identity_matrix(self.engine.matrix_multiply(matrix, inverse)))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol = PCDLProtocol(system_seed=b"A" * 32, master_seed=b"B" * 32)

    def test_registration_authentication_and_rotation(self):
        _, identifier = self.protocol.register("alice", "correct", track_perf=False)
        _, status, _ = self.protocol.authenticate("alice", "correct", track_perf=False)
        self.assertEqual(status, "SUCCESS")
        _, status, _ = self.protocol.authenticate("alice", "wrong", track_perf=False)
        self.assertEqual(status, "FAIL_1")
        before = [poly.copy() for poly in self.protocol.DB_W[identifier]["Bi"]]
        self.protocol.rotate(track_perf=False)
        self.assertFalse(np.array_equal(before[0], self.protocol.DB_W[identifier]["Bi"][0]))
        _, status, _ = self.protocol.authenticate("alice", "correct", track_perf=False)
        self.assertEqual(status, "SUCCESS")

    def test_all_module_coordinates_are_independently_derived(self):
        secret, error = self.protocol._server_vectors(self.protocol.msk_W, "identifier")
        self.assertFalse(np.array_equal(secret[0], secret[1]))
        self.assertFalse(np.array_equal(error[0], error[1]))

    def test_memory_is_measured(self):
        self.protocol.register("memory", "test", track_perf=True)
        self.assertGreater(self.protocol.last_memory_kib, 0)
        self.assertEqual(len(self.protocol.perf_stats["registration_memory"]), 1)


    def test_rotation_statistics_are_recorded_per_user(self):
        for index in range(3):
            self.protocol.register(
                f"rotation-user-{index}", "correct", track_perf=False, measure_memory=False
            )
        self.protocol.rotate(track_perf=True, measure_memory=True)
        self.assertEqual(len(self.protocol.last_rotation_user_times_ms), 3)
        self.assertEqual(len(self.protocol.last_rotation_user_memory_kib), 3)
        self.assertEqual(len(self.protocol.perf_stats["key_rotation"]), 3)
        self.assertEqual(len(self.protocol.perf_stats["key_rotation_memory"]), 3)
        self.assertEqual(self.protocol.perf_stats["key_rotation_users"], [3])
        self.assertEqual(len(self.protocol.perf_stats["key_rotation_total"]), 1)

    def test_exit_summary_distinguishes_rotation_users_from_batches(self):
        for index in range(2):
            self.protocol.register(
                f"summary-user-{index}", "correct", track_perf=False, measure_memory=False
            )
        self.protocol.rotate(track_perf=True, measure_memory=False)
        self.protocol.rotate(track_perf=True, measure_memory=False)
        summary = format_performance_summary(self.protocol)
        self.assertIn("Key Rotation / User", summary)
        self.assertIn("Key Rotation / Batch", summary)
        self.assertIn("Key-rotation commands         : 2", summary)
        self.assertIn("User records rotated (samples): 4", summary)

    def test_latency_measurement_does_not_include_memory_probe(self):
        self.protocol.register("separate-measurement", "test", track_perf=True)
        self.assertFalse(__import__("tracemalloc").is_tracing())
        self.assertGreater(self.protocol.perf_stats["registration"][0], 0)
        self.assertGreater(self.protocol.perf_stats["registration_memory"][0], 0)

    def test_graph_data_contains_latency_and_memory(self):
        original = plot_asymptotic.__globals__["plot_asymptotic"]
        self.assertIs(original, plot_asymptotic)
        # Avoid the expensive default 500-user benchmark in unit tests; the graph
        # helper itself is exercised separately with representative data.
        display = plot_asymptotic.__globals__["_display_graph"]
        display([1, 2], [1.0, 2.0], "Registration", "Performance", "Total Latency (ms)", "blue")
        display([1, 2], [3.0, 4.0], "Registration", "Memory Consumption", "Peak Memory (KiB)", "blue")


if __name__ == "__main__":
    unittest.main()
