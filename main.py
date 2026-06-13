# -*- coding: utf-8 -*-
"""Colab-compatible prototype of server-aided password authentication.

Google Colab users should download and run the complete ``main.ipynb`` file.
Red/green lines on GitHub are a review diff and must not be copied into Colab.
The complete ``main.py`` file is the optional local/script entry point.

The lattice layer follows the ML-KEM-512 ring parameters and the NTT/base-
multiplication definitions from NIST FIPS 203.  This remains a research
prototype of the surrounding password protocol; it is not an ML-KEM
implementation or a substitute for an independently reviewed product.
"""

import gc
import hashlib
import secrets
import time
import tracemalloc
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np


IMPLEMENTATION_VERSION = "2026.06-per-user-rotation-v2"


# -----------------------------------------------------------------------------
# 1. ML-KEM-512-ALIGNED RING AND NTT ENGINE
# -----------------------------------------------------------------------------
class LatticeEngine:
    """Arithmetic in R_q = Z_q[X]/(X^256 + 1), using the FIPS 203 NTT."""

    def __init__(self):
        self.n = 256
        self.q = 3329
        self.zeta = 17
        self.rank = 2
        self.eta1 = 3
        self.seed_bytes = 32
        self.zetas = np.array(
            [pow(self.zeta, self._bit_reverse_7(i), self.q) for i in range(128)],
            dtype=np.int64,
        )

    @staticmethod
    def _bit_reverse_7(value):
        return int(f"{value:07b}"[::-1], 2)

    @staticmethod
    def encode_fields(*fields):
        """Unambiguously length-prefix fields before hashing or XOF use."""
        encoded = bytearray()
        for field in fields:
            if isinstance(field, str):
                field = field.encode("utf-8")
            elif isinstance(field, np.ndarray):
                field = np.asarray(field, dtype="<i8").tobytes()
            elif not isinstance(field, (bytes, bytearray)):
                field = str(field).encode("utf-8")
            field = bytes(field)
            encoded.extend(len(field).to_bytes(4, "little"))
            encoded.extend(field)
        return bytes(encoded)

    def shake128(self, domain, *fields, output_bytes):
        material = self.encode_fields(b"PCDL-v2", domain, *fields)
        return hashlib.shake_128(material).digest(output_bytes)

    def ntt(self, poly):
        """FIPS 203 NTT (Algorithm 9), represented as 128 quadratic pairs."""
        r = np.asarray(poly, dtype=np.int64).copy() % self.q
        if r.shape != (self.n,):
            raise ValueError(f"polynomial must contain exactly {self.n} coefficients")
        k, length = 1, 128
        while length >= 2:
            for start in range(0, self.n, 2 * length):
                zeta = int(self.zetas[k])
                k += 1
                left = r[start : start + length].copy()
                right = r[start + length : start + 2 * length].copy()
                product = (zeta * right) % self.q
                r[start : start + length] = (left + product) % self.q
                r[start + length : start + 2 * length] = (left - product) % self.q
            length //= 2
        return r

    def _pair_gamma(self, pair_index):
        gamma = int(self.zetas[64 + pair_index // 2])
        return gamma if pair_index % 2 == 0 else (-gamma) % self.q

    def multiply_ntt(self, left, right):
        """FIPS 203 NTT-domain base multiplication (Algorithm 11)."""
        left = np.asarray(left, dtype=np.int64)
        right = np.asarray(right, dtype=np.int64)
        if left.shape != (self.n,) or right.shape != (self.n,):
            raise ValueError("NTT operands must contain exactly 256 coefficients")
        out = np.empty(self.n, dtype=np.int64)
        for pair in range(128):
            i = 2 * pair
            gamma = self._pair_gamma(pair)
            a0, a1 = int(left[i]), int(left[i + 1])
            b0, b1 = int(right[i]), int(right[i + 1])
            out[i] = (a0 * b0 + gamma * a1 * b1) % self.q
            out[i + 1] = (a0 * b1 + a1 * b0) % self.q
        return out

    def add(self, left, right):
        return (np.asarray(left, dtype=np.int64) + np.asarray(right, dtype=np.int64)) % self.q

    def subtract(self, left, right):
        return (np.asarray(left, dtype=np.int64) - np.asarray(right, dtype=np.int64)) % self.q

    def cbd_sample(self, seed, domain, nonce, eta=None):
        """Sample one polynomial from CBD_eta using SHAKE-128.

        ML-KEM-512 uses eta1=3 for key-generation secret and error vectors.
        This prototype uses that same predefined domain for s_i, e_i, and h_i.
        """
        eta = self.eta1 if eta is None else eta
        byte_count = (self.n * 2 * eta + 7) // 8
        data = self.shake128(domain, seed, nonce, output_bytes=byte_count)
        value = int.from_bytes(data, "little")
        mask = (1 << eta) - 1
        poly = np.empty(self.n, dtype=np.int64)
        for i in range(self.n):
            a = (value >> (2 * eta * i)) & mask
            b = (value >> (2 * eta * i + eta)) & mask
            poly[i] = (a.bit_count() - b.bit_count()) % self.q
        return poly

    def uniform_poly(self, seed, domain, nonce):
        """SHAKE-128 rejection sampling of uniform coefficients in Z_q."""
        coefficients = []
        output_length = 672
        while len(coefficients) < self.n:
            data = self.shake128(domain, seed, nonce, output_bytes=output_length)
            coefficients.clear()
            for i in range(0, len(data) - 2, 3):
                d1 = data[i] | ((data[i + 1] & 0x0F) << 8)
                d2 = (data[i + 1] >> 4) | (data[i + 2] << 4)
                if d1 < self.q:
                    coefficients.append(d1)
                if d2 < self.q and len(coefficients) < self.n:
                    coefficients.append(d2)
                if len(coefficients) == self.n:
                    break
            output_length *= 2
        return np.array(coefficients, dtype=np.int64)

    def one_ntt(self):
        one = np.zeros(self.n, dtype=np.int64)
        one[0] = 1
        return self.ntt(one)

    def invert_ntt(self, value):
        """Invert a unit of R_q in the 128 quadratic NTT factors."""
        value = np.asarray(value, dtype=np.int64)
        inverse = np.empty(self.n, dtype=np.int64)
        for pair in range(128):
            i = 2 * pair
            gamma = self._pair_gamma(pair)
            x, y = int(value[i]), int(value[i + 1])
            norm = (x * x - gamma * y * y) % self.q
            if norm == 0:
                raise ValueError("polynomial is not a unit in R_q")
            norm_inverse = pow(int(norm), -1, self.q)
            inverse[i] = x * norm_inverse % self.q
            inverse[i + 1] = -y * norm_inverse % self.q
        return inverse

    def matrix_multiply(self, left, right):
        result = [[np.zeros(self.n, dtype=np.int64) for _ in range(2)] for _ in range(2)]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    result[i][j] = self.add(
                        result[i][j], self.multiply_ntt(left[i][k], right[k][j])
                    )
        return result

    def matrix_vector_multiply(self, matrix, vector):
        result = []
        for i in range(self.rank):
            accumulator = np.zeros(self.n, dtype=np.int64)
            for j in range(self.rank):
                accumulator = self.add(
                    accumulator, self.multiply_ntt(matrix[i][j], vector[j])
                )
            result.append(accumulator)
        return result

    def generate_invertible_matrix(self, seed):
        """Sample a uniform 2x2 matrix and reject until its determinant is a unit."""
        for attempt in range(1_000_000):
            matrix = []
            for row in range(self.rank):
                matrix_row = []
                for column in range(self.rank):
                    poly = self.uniform_poly(
                        seed, b"matrix-A", self.encode_fields(attempt, row, column)
                    )
                    matrix_row.append(self.ntt(poly))
                matrix.append(matrix_row)
            determinant = self.subtract(
                self.multiply_ntt(matrix[0][0], matrix[1][1]),
                self.multiply_ntt(matrix[0][1], matrix[1][0]),
            )
            try:
                determinant_inverse = self.invert_ntt(determinant)
            except ValueError:
                continue
            inverse = [
                [
                    self.multiply_ntt(determinant_inverse, matrix[1][1]),
                    self.multiply_ntt(determinant_inverse, (-matrix[0][1]) % self.q),
                ],
                [
                    self.multiply_ntt(determinant_inverse, (-matrix[1][0]) % self.q),
                    self.multiply_ntt(determinant_inverse, matrix[0][0]),
                ],
            ]
            return matrix, inverse, attempt + 1
        raise RuntimeError("failed to generate an invertible matrix")

    def is_identity_matrix(self, matrix):
        one, zero = self.one_ntt(), np.zeros(self.n, dtype=np.int64)
        return all(
            np.array_equal(matrix[i][j] % self.q, one if i == j else zero)
            for i in range(2)
            for j in range(2)
        )


# -----------------------------------------------------------------------------
# 2. PROTOCOL LOGIC
# -----------------------------------------------------------------------------
class PCDLProtocol:
    def __init__(self, system_seed=None, master_seed=None):
        self.eng = LatticeEngine()
        self.q, self.l = self.eng.q, self.eng.rank
        self.limit, self.window, self.lock_duration = 5, 60, 30
        self.Wj = b"WEBSITE_TAG_ALPHA"
        self.system_seed = system_seed or secrets.token_bytes(self.eng.seed_bytes)
        self.msk_W = master_seed or secrets.token_bytes(self.eng.seed_bytes)
        self.A, self.A_inverse, self.matrix_attempts = self.eng.generate_invertible_matrix(
            self.system_seed
        )
        if not self.eng.is_identity_matrix(self.eng.matrix_multiply(self.A, self.A_inverse)):
            raise RuntimeError("generated matrix A failed its invertibility check")
        self.DB_W = {}
        self.DB_S = {}
        self.last_memory_kib = 0.0
        self.last_rotation_total_ms = 0.0
        self.last_rotation_user_times_ms = []
        self.last_rotation_user_memory_kib = []
        self.perf_stats = {
            "registration": [],
            "authentication": [],
            # Key-rotation samples are recorded per updated user, not per batch.
            "key_rotation": [],
            "registration_memory": [],
            "authentication_memory": [],
            "key_rotation_memory": [],
            "key_rotation_total": [],
            "key_rotation_users": [],
        }
        self._warm_up_measurement_paths()

    def _identifier(self, username):
        return self.eng.shake128(
            b"user-identifier", username, self.Wj, output_bytes=32
        ).hex()

    def _password_vector(self, username, password, salt):
        seed = self.eng.shake128(
            b"password-seed", password, username, salt, self.Wj, output_bytes=32
        )
        return [
            self.eng.cbd_sample(seed, b"password-h", coordinate, self.eng.eta1)
            for coordinate in range(self.l)
        ]

    def _server_vectors(self, master_seed, identifier):
        secret = [
            self.eng.cbd_sample(master_seed, b"server-s", self.eng.encode_fields(identifier, i))
            for i in range(self.l)
        ]
        error = [
            self.eng.cbd_sample(master_seed, b"server-e", self.eng.encode_fields(identifier, i))
            for i in range(self.l)
        ]
        return secret, error

    def _verifier(self, username, password, salt, identifier, master_seed):
        password_vector = self._password_vector(username, password, salt)
        secret, error = self._server_vectors(master_seed, identifier)
        combined_ntt = [
            self.eng.ntt(self.eng.add(password_vector[i], secret[i])) for i in range(self.l)
        ]
        error_ntt = [self.eng.ntt(poly) for poly in error]
        product = self.eng.matrix_vector_multiply(self.A, combined_ntt)
        return [self.eng.add(product[i], error_ntt[i]) for i in range(self.l)]

    def _warm_up_measurement_paths(self):
        """Run one unmeasured verifier calculation to remove first-call bias."""
        self._verifier(
            "__pcdl_warmup__",
            "warmup",
            bytes(self.eng.seed_bytes),
            "0" * 64,
            self.msk_W,
        )

    @staticmethod
    def _time_call(operation):
        """Time an operation without tracemalloc or cyclic-GC measurement noise."""
        tracing = tracemalloc.is_tracing()
        if tracing:
            tracemalloc.stop()
        gc_was_enabled = gc.isenabled()
        if gc_was_enabled:
            gc.disable()
        try:
            started = time.perf_counter_ns()
            result = operation()
            elapsed = (time.perf_counter_ns() - started) / 1_000_000_000.0
        finally:
            if gc_was_enabled:
                gc.enable()
        return result, elapsed

    @staticmethod
    def _measure_peak_memory(operation):
        """Measure a separate probe so tracing overhead never enters latency."""
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        tracemalloc.start()
        tracemalloc.reset_peak()
        baseline = tracemalloc.get_traced_memory()[0]
        try:
            operation()
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return max(0, peak - baseline) / 1024.0

    def register(self, un, pwd, track_perf=True, measure_memory=True):
        ti = self._identifier(un)
        salt = secrets.token_bytes(self.eng.seed_bytes)

        def registration_core():
            Bi = self._verifier(un, pwd, salt, ti, self.msk_W)
            self.DB_W[ti] = {"un": un, "salt": salt, "Bi": Bi}
            self.DB_S[ti] = {"count": 0, "last_fail": 0, "lock_until": 0}
            return Bi

        _, elapsed = self._time_call(registration_core)
        memory_kib = 0.0
        if measure_memory:
            # Probe the same verifier workload without replacing the stored record.
            memory_kib = self._measure_peak_memory(
                lambda: self._verifier(un, pwd, salt, ti, self.msk_W)
            )
            self.last_memory_kib = memory_kib
        if track_perf:
            self.perf_stats["registration"].append(elapsed * 1000)
            if measure_memory:
                self.perf_stats["registration_memory"].append(memory_kib)
        return elapsed, ti

    def authenticate(self, un, pwd_in, track_perf=True, measure_memory=True):
        ti = self._identifier(un)
        if ti not in self.DB_W:
            return None, "USER_NOT_FOUND", 0
        state = self.DB_S[ti]
        curr_time = time.time()
        if curr_time < state["lock_until"]:
            return None, f"LOCKED_OUT_{int(state['lock_until'] - curr_time)}s", 0

        record = self.DB_W[ti]

        def authentication_core():
            calculated = self._verifier(un, pwd_in, record["salt"], ti, self.msk_W)
            valid = all(
                secrets.compare_digest(
                    np.asarray(calculated[i], dtype="<u2").tobytes(),
                    np.asarray(record["Bi"][i], dtype="<u2").tobytes(),
                )
                for i in range(self.l)
            )
            return calculated, valid

        (Bi_calc, is_valid), elapsed = self._time_call(authentication_core)
        memory_kib = 0.0
        if measure_memory:
            memory_kib = self._measure_peak_memory(authentication_core)
            self.last_memory_kib = memory_kib
        if track_perf:
            self.perf_stats["authentication"].append(elapsed * 1000)
            if measure_memory:
                self.perf_stats["authentication_memory"].append(memory_kib)

        if is_valid:
            self.DB_S[ti] = {"count": 0, "last_fail": 0, "lock_until": 0}
            return Bi_calc, "SUCCESS", elapsed

        if curr_time - state["last_fail"] > self.window:
            state["count"] = 1
        else:
            state["count"] += 1
        state["last_fail"] = curr_time
        if state["count"] >= self.limit:
            state["lock_until"] = curr_time + self.lock_duration
        return Bi_calc, f"FAIL_{state['count']}", elapsed

    def _rotate_record(self, data, identifier, old_msk, new_msk):
        old_s, old_e = self._server_vectors(old_msk, identifier)
        new_s, new_e = self._server_vectors(new_msk, identifier)
        old_s_ntt = [self.eng.ntt(poly) for poly in old_s]
        new_s_ntt = [self.eng.ntt(poly) for poly in new_s]
        old_e_ntt = [self.eng.ntt(poly) for poly in old_e]
        new_e_ntt = [self.eng.ntt(poly) for poly in new_e]
        masked_old = [
            self.eng.subtract(data["Bi"][i], old_e_ntt[i]) for i in range(self.l)
        ]
        combined_old = self.eng.matrix_vector_multiply(self.A_inverse, masked_old)
        combined_new = [
            self.eng.add(
                self.eng.subtract(combined_old[i], old_s_ntt[i]), new_s_ntt[i]
            )
            for i in range(self.l)
        ]
        remasked = self.eng.matrix_vector_multiply(self.A, combined_new)
        data["Bi"] = [
            self.eng.add(remasked[i], new_e_ntt[i]) for i in range(self.l)
        ]

    def rotate(self, track_perf=True, measure_memory=True):
        """Rotate the server seed and record latency/memory for each user."""
        total_started = time.perf_counter_ns()
        old_msk = self.msk_W
        new_msk = secrets.token_bytes(self.eng.seed_bytes)
        old_entries = deepcopy(self.DB_W)
        per_user_times_ms = []

        for ti, data in self.DB_W.items():
            _, elapsed = self._time_call(
                lambda data=data, ti=ti: self._rotate_record(data, ti, old_msk, new_msk)
            )
            per_user_times_ms.append(elapsed * 1000)
        total_elapsed = (time.perf_counter_ns() - total_started) / 1_000_000_000.0

        per_user_memory_kib = []
        if measure_memory:
            for ti, old_data in old_entries.items():
                probe = deepcopy(old_data)
                per_user_memory_kib.append(
                    self._measure_peak_memory(
                        lambda probe=probe, ti=ti: self._rotate_record(
                            probe, ti, old_msk, new_msk
                        )
                    )
                )

        self.msk_W = new_msk
        self.last_rotation_total_ms = total_elapsed * 1000
        self.last_rotation_user_times_ms = per_user_times_ms
        self.last_rotation_user_memory_kib = per_user_memory_kib
        self.last_memory_kib = float(np.mean(per_user_memory_kib)) if per_user_memory_kib else 0.0
        if track_perf:
            self.perf_stats["key_rotation"].extend(per_user_times_ms)
            self.perf_stats["key_rotation_memory"].extend(per_user_memory_kib)
            self.perf_stats["key_rotation_total"].append(total_elapsed * 1000)
            self.perf_stats["key_rotation_users"].append(len(per_user_times_ms))
        return old_msk, new_msk, old_entries, total_elapsed


# -----------------------------------------------------------------------------
# 3. GRAPH GENERATOR (LATENCY AND MEMORY) - NON-BLOCKING
# -----------------------------------------------------------------------------
def _display_graph(batch_sizes, values, phase_name, metric, ylabel, color):
    plt.figure(figsize=(10, 5))
    plt.plot(
        batch_sizes,
        values,
        marker="s",
        linestyle="-",
        linewidth=2,
        markersize=8,
        color=color,
    )
    plt.title(f"Asymptotic {phase_name} {metric}", fontweight="bold", fontsize=14)
    plt.xlabel("Number of Users", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)


def plot_asymptotic(phase_name, protocol_instance, repetitions=3):
    """Plot stable median latency and memory measurements.

    Registration and authentication retain total batch latency. Key rotation is
    normalized per updated user, because one rotation call processes the whole
    database. Each latency point is the median of repeated, independently
    prepared runs, and memory is measured in a separate pass.
    """
    print(f"\n{'=' * 60}")
    print(f"Generating Stable Latency and Memory Graphs for {phase_name}...")
    print(f"{'=' * 60}")
    batch_sizes = [1, 10, 50, 100, 250, 500]
    times, memory = [], []

    def prepared_protocol(user_count):
        protocol = PCDLProtocol(
            system_seed=protocol_instance.system_seed,
            master_seed=protocol_instance.msk_W,
        )
        if phase_name in {"Authentication", "Key Rotation"}:
            prefix = "auth_bench" if phase_name == "Authentication" else "rot_bench"
            for index in range(user_count):
                protocol.register(
                    f"{prefix}_{index}", "pass", track_perf=False, measure_memory=False
                )
        return protocol

    for user_count in batch_sizes:
        latency_samples = []
        for repeat in range(repetitions):
            temp_proto = prepared_protocol(user_count)
            started = time.perf_counter_ns()
            if phase_name == "Registration":
                for index in range(user_count):
                    temp_proto.register(
                        f"reg_bench_{repeat}_{index}",
                        "pass",
                        track_perf=False,
                        measure_memory=False,
                    )
                measured_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            elif phase_name == "Authentication":
                for index in range(user_count):
                    temp_proto.authenticate(
                        f"auth_bench_{index}",
                        "pass",
                        track_perf=False,
                        measure_memory=False,
                    )
                measured_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            elif phase_name == "Key Rotation":
                temp_proto.rotate(track_perf=False, measure_memory=False)
                measured_ms = float(np.mean(temp_proto.last_rotation_user_times_ms))
            else:
                raise ValueError(f"unknown phase: {phase_name}")
            latency_samples.append(measured_ms)
        times.append(float(np.median(latency_samples)))

        memory_proto = prepared_protocol(user_count)
        if phase_name == "Registration":
            memory_samples = []
            for index in range(user_count):
                memory_proto.register(
                    f"memory_reg_{index}", "pass", track_perf=False, measure_memory=True
                )
                memory_samples.append(memory_proto.last_memory_kib)
            memory.append(float(np.sum(memory_samples)))
        elif phase_name == "Authentication":
            memory_samples = []
            for index in range(user_count):
                memory_proto.authenticate(
                    f"auth_bench_{index}", "pass", track_perf=False, measure_memory=True
                )
                memory_samples.append(memory_proto.last_memory_kib)
            memory.append(float(np.sum(memory_samples)))
        else:
            memory_proto.rotate(track_perf=False, measure_memory=True)
            memory.append(float(np.mean(memory_proto.last_rotation_user_memory_kib)))

    color_map = {"Registration": "blue", "Authentication": "green", "Key Rotation": "red"}
    color = color_map.get(phase_name, "blue")
    latency_ylabel = (
        "Median Latency per User (ms)"
        if phase_name == "Key Rotation"
        else "Median Total Latency (ms)"
    )
    memory_ylabel = (
        "Mean Peak Memory per User (KiB)"
        if phase_name == "Key Rotation"
        else "Cumulative Peak Memory (KiB)"
    )
    _display_graph(batch_sizes, times, phase_name, "Performance", latency_ylabel, color)
    _display_graph(batch_sizes, memory, phase_name, "Memory Consumption", memory_ylabel, color)
    print("\n[Latency and memory graphs displayed. You can close them or leave them open.]\n")
    return {
        "batch_sizes": batch_sizes,
        "latency_ms": times,
        "peak_memory_kib": memory,
        "latency_basis": "per_user" if phase_name == "Key Rotation" else "total_batch",
    }


def format_performance_summary(protocol):
    """Return an explicit per-operation and per-rotation-batch summary."""
    rows = []
    for label, key in (
        ("Registration / User", "registration"),
        ("Authentication / User", "authentication"),
        ("Key Rotation / User", "key_rotation"),
    ):
        timings = protocol.perf_stats[key]
        memories = protocol.perf_stats[f"{key}_memory"]
        if timings:
            rows.append(
                (
                    label,
                    len(timings),
                    float(np.mean(timings)),
                    float(np.min(timings)),
                    float(np.max(timings)),
                    float(np.std(timings)),
                    float(np.mean(memories)) if memories else 0.0,
                )
            )

    batch_timings = protocol.perf_stats["key_rotation_total"]
    if batch_timings:
        rows.append(
            (
                "Key Rotation / Batch",
                len(batch_timings),
                float(np.mean(batch_timings)),
                float(np.min(batch_timings)),
                float(np.max(batch_timings)),
                float(np.std(batch_timings)),
                0.0,
            )
        )

    lines = [
        "{:<23} {:>8} {:>11} {:>11} {:>11} {:>11} {:>14}".format(
            "Metric basis", "Samples", "Avg ms", "Min ms", "Max ms", "Std ms", "Avg Mem KiB"
        ),
        "-" * 96,
    ]
    lines.extend(
        "{:<23} {:>8} {:>11.4f} {:>11.4f} {:>11.4f} {:>11.4f} {:>14.2f}".format(*row)
        for row in rows
    )
    rotation_batches = len(protocol.perf_stats["key_rotation_total"])
    users_rotated = len(protocol.perf_stats["key_rotation"])
    lines.extend(
        [
            "",
            "Operation counts:",
            f"  Registrations performed       : {len(protocol.perf_stats['registration'])}",
            f"  Authentications performed     : {len(protocol.perf_stats['authentication'])}",
            f"  Key-rotation commands         : {rotation_batches}",
            f"  User records rotated (samples): {users_rotated}",
            "",
            "Key Rotation / User is the per-record latency requested; ",
            "Key Rotation / Batch is the total time for one rotation command.",
        ]
    )
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# 4. MAIN INTERACTIVE EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    plt.ion()
    proto = PCDLProtocol()

    print("\n" + "=" * 70)
    print(" " * 15 + "PCDL AUTHENTICATION PROTOCOL")
    print(f" " * 13 + f"Build: {IMPLEMENTATION_VERSION}")
    print("=" * 70)
    print("\n--- SYSTEM PARAMETERS ---")
    print(f"  Modulus (q)              : {proto.q}")
    print(f"  Polynomial Degree (n)    : {proto.eng.n}")
    print(f"  Module Rank (l)          : {proto.l}")
    print(f"  CBD Parameter (eta1)     : {proto.eng.eta1}")
    print(f"  Matrix A Invertible      : Yes ({proto.matrix_attempts} attempt(s))")
    print(f"  Failed Attempt Limit     : {proto.limit}")
    print(f"  Time Window for Fails    : {proto.window} seconds")
    print(f"  Account Lock Duration    : {proto.lock_duration} seconds")
    print("  Seeds                    : 256-bit CSPRNG values (not displayed)")
    print("=" * 70)

    print("\n" + "=" * 70)
    print(" " * 20 + "REGISTRATION PHASE")
    print("=" * 70)
    print("\nPlease register at least 3 users to begin.\n")
    for i in range(3):
        print(f"\n[Registering User {i + 1}/3]")
        username = input("  Username: ").strip()
        password = input("  Password: ").strip()
        elapsed, identifier = proto.register(username, password)
        print("\n  ✓ Registration Successful!")
        print(f"  → User Identifier: {identifier[:16]}...")
        print(f"  → Registration Time: {elapsed * 1000:.4f} ms")
        print(f"  → Peak Memory: {proto.last_memory_kib:.2f} KiB")
        print(f"  → Bi[0] (first 10): {proto.DB_W[identifier]['Bi'][0][:10]}")
        print(f"  → Bi[1] (first 10): {proto.DB_W[identifier]['Bi'][1][:10]}")

    plot_asymptotic("Registration", proto)

    while True:
        print("\n" + "=" * 70)
        print(" " * 24 + "MAIN MENU")
        print("=" * 70)
        print("  1. Register New User")
        print("  2. Authenticate User")
        print("  3. Rotate Server Master Seed")
        print("  4. Exit and Show Performance Summary")
        command = input("\n  Select option (1-4): ").strip()

        if command == "1":
            username = input("  Username: ").strip()
            password = input("  Password: ").strip()
            elapsed, identifier = proto.register(username, password)
            print(f"\n  ✓ Registered in {elapsed * 1000:.4f} ms")
            print(f"  → Peak Memory: {proto.last_memory_kib:.2f} KiB")
            plot_asymptotic("Registration", proto)

        elif command == "2":
            username = input("  Username: ").strip()
            password = input("  Password: ").strip()
            calculated, status, elapsed = proto.authenticate(username, password)
            print(f"\n  → Status: {status}")
            if elapsed:
                print(f"  → Authentication Time: {elapsed * 1000:.4f} ms")
                print(f"  → Peak Memory: {proto.last_memory_kib:.2f} KiB")
            if calculated is not None:
                print(f"  → Calculated Bi[0] (first 10): {calculated[0][:10]}")
                print(f"  → Calculated Bi[1] (first 10): {calculated[1][:10]}")
            plot_asymptotic("Authentication", proto)

        elif command == "3":
            old_seed, new_seed, old_entries, elapsed = proto.rotate()
            print("\n  ✓ KEY ROTATION COMPLETED!")
            print(f"  → Total Rotation Time: {elapsed * 1000:.4f} ms")
            if proto.last_rotation_user_times_ms:
                print(
                    f"  → Average Per-User Time: "
                    f"{np.mean(proto.last_rotation_user_times_ms):.4f} ms"
                )
                print(
                    f"  → Average Per-User Peak Memory: "
                    f"{np.mean(proto.last_rotation_user_memory_kib):.2f} KiB"
                )
            print(f"  → Users Updated: {len(old_entries)}")
            print(f"  → Old Seed Fingerprint: {hashlib.shake_128(old_seed).hexdigest(8)}")
            print(f"  → New Seed Fingerprint: {hashlib.shake_128(new_seed).hexdigest(8)}")
            plot_asymptotic("Key Rotation", proto)

        elif command == "4":
            print("\n" + "=" * 70)
            print(" " * 20 + "PERFORMANCE SUMMARY")
            print("=" * 70)
            print("\n" + format_performance_summary(proto))
            print("\n" + "=" * 70)
            print(" " * 25 + "Exiting Protocol")
            print("=" * 70)
            plt.close("all")
            break

        else:
            print("\n  ✗ Invalid option. Please select 1-4.")
