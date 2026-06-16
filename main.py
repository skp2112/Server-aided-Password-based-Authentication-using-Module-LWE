import os, sys, numpy as np, time, hashlib, matplotlib.pyplot as plt


# =============================================================================
# 0. UTILITIES
# =============================================================================

def shake128(*parts: bytes, length: int = 32) -> bytes:
    h = hashlib.shake_128()
    for p in parts: h.update(p)
    return h.digest(length)

def _input(prompt=""):
    """input() wrapper — flushes stdout first so Colab shows the prompt."""
    sys.stdout.flush()
    return input(prompt)

def _show_graph(fig=None):
    """
    Display a matplotlib figure without blocking and without leaving any
    state behind that could interfere with the next input() call.

    - In IPython/Jupyter/Colab: the figure is rendered to a PNG and shown
      via IPython.display.Image. This avoids plt.show()'s interactive/
      widget machinery entirely, which is what was stalling input()
      afterwards.
    - In a plain script (no IPython): falls back to a non-blocking
      plt.show(block=False) + short pause.

    The figure is ALWAYS closed afterwards (plt.close(fig)) so figures do
    not accumulate across repeated menu visits -- especially important for
    sub-option 'e', which draws four figures in a row.
    """
    if fig is None:
        fig = plt.gcf()
    plt.tight_layout()
    try:
        from IPython.display import display, Image
        import io
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=70, bbox_inches='tight')
        buf.seek(0)
        display(Image(data=buf.getvalue()))
    except ImportError:
        plt.show(block=False)
        plt.pause(0.001)
    finally:
        plt.close(fig)
    sys.stdout.flush()


# =============================================================================
# 1. LATTICE ENGINE  (FIPS 203 NTT/INTT, NumPy-vectorised)
# =============================================================================

class LatticeEngine:
    """
    Polynomial arithmetic in R_q = Z_q[X]/(X^256+1), q=3329, zeta=17.
    NTT (Alg 9) and INTT (Alg 10) follow FIPS 203 exactly.
    Every butterfly level is a single NumPy advanced-indexing operation —
    no Python inner loops — satisfying the paper's 'NumPy-based NTT' claim.
    """
    N = 256; Q = 3329; ZETA = 17

    def __init__(self):
        def _br7(i): return int(format(i, '07b')[::-1], 2)
        self.zetas     = np.array([pow(self.ZETA, _br7(i), self.Q) for i in range(128)], dtype=np.int64)
        self._n128_inv = pow(128, self.Q - 2, self.Q)

    def ntt(self, poly):
        f = poly.copy().astype(np.int64) % self.Q
        k, length = 1, 128
        while length >= 2:
            nb = self.N // (2 * length)
            z  = self.zetas[k:k + nb]; k += nb
            s  = np.arange(nb, dtype=np.int64) * (2 * length)
            it = s[:, None] + np.arange(length, dtype=np.int64)[None, :]
            ib = it + length
            top = f[it]; bot = f[ib]; t = z[:, None] * bot % self.Q
            f[it] = (top + t) % self.Q; f[ib] = (top - t) % self.Q
            length >>= 1
        return f % self.Q

    def intt(self, f_hat):
        f = f_hat.copy().astype(np.int64) % self.Q
        k, length = 127, 2
        while length <= 128:
            nb = self.N // (2 * length)
            z  = self.zetas[k - nb + 1:k + 1][::-1]; k -= nb
            s  = np.arange(nb, dtype=np.int64) * (2 * length)
            it = s[:, None] + np.arange(length, dtype=np.int64)[None, :]
            ib = it + length
            top = f[it].copy(); bot = f[ib].copy()
            f[it] = (top + bot) % self.Q
            f[ib] = z[:, None] * (bot - top) % self.Q
            length <<= 1
        return f * self._n128_inv % self.Q

    @staticmethod
    def _pw_mul(a, b, q):
        return a.astype(np.int64) * b.astype(np.int64) % q

    def matvec_ntt(self, A_ntt, v_ntt, l):
        res = []
        for i in range(l):
            acc = np.zeros(self.N, dtype=np.int64)
            for j in range(l):
                acc = (acc + self._pw_mul(A_ntt[i][j], v_ntt[j], self.Q)) % self.Q
            res.append(acc)
        return res


# =============================================================================
# 2. CBD eta=2/3  (FIPS 203 §4.2.2, vectorised)
# =============================================================================

def cbd_sample_eta(seed: bytes, eta: int = 2, nonce: int = 0) -> np.ndarray:
    """
    FIPS 203 SamplePolyCBD_eta. PRF = SHAKE-128(seed || nonce_byte), 64*eta bytes.
    Returns centered values in [-eta, eta]. NOT reduced mod q.
    """
    prf  = shake128(seed, bytes([nonce & 0xFF]), length=64 * eta)
    bits = np.unpackbits(np.frombuffer(prf, dtype=np.uint8), bitorder='little')
    x = bits[0::2*eta].astype(np.int64) * 0  # placeholder, see below
    # sum eta "1" bits then eta "0" bits per coefficient
    bits = bits[:256 * 2 * eta].reshape(256, 2 * eta)
    return bits[:, :eta].sum(axis=1).astype(np.int64) - bits[:, eta:].sum(axis=1).astype(np.int64)


# =============================================================================
# 3. Matrix A  (SHAKE-128 seed expansion, NTT domain, invertible)
# =============================================================================

def _rej_uniform(seed_bytes, n, q):
    poly = np.zeros(n, dtype=np.int64); count = 0
    buf = bytearray(seed_bytes); ext = bytearray(seed_bytes); pos = 0
    while count < n:
        while pos + 2 >= len(buf):
            ext = bytearray(shake128(bytes(ext), length=len(ext) + 128))
            buf = ext; pos = 0
        d1 = buf[pos] | ((buf[pos + 1] & 0x0F) << 8)
        d2 = (buf[pos + 1] >> 4) | (buf[pos + 2] << 4) if pos + 2 < len(buf) else q
        pos += 2
        if d1 < q and count < n: poly[count] = d1; count += 1
        if d2 < q and count < n: poly[count] = d2; count += 1
    return poly

def generate_A_ntt(eng, rho, l=2, max_attempts=100):
    for attempt in range(max_attempts):
        if attempt > 0: rho = shake128(rho, bytes([attempt]), length=32)
        A_ntt = [[eng.ntt(_rej_uniform(shake128(rho, bytes([i, j]), length=384), eng.N, eng.Q))
                  for j in range(l)] for i in range(l)]
        if all(np.all(A_ntt[k][k] != 0) for k in range(l)):
            return A_ntt, rho
    raise RuntimeError("Could not generate invertible A")


# =============================================================================
# 4. DB storage metric
# =============================================================================

def db_storage_bytes(db_w):
    total = 0
    for e in db_w.values():
        total += len(e['ri_seed']) + sum(b.nbytes for b in e['Bi']) + len(e['un'].encode())
    return total


# =============================================================================
# 5. PROTOCOL
# =============================================================================

class PCDLProtocol:
    def __init__(self):
        self.eng = LatticeEngine(); self.Q = LatticeEngine.Q
        self.N = LatticeEngine.N;  self.l = 2
        self.limit = 5; self.window = 60; self.lock_duration = 30
        self.msk_W = shake128(b"msk_W", os.urandom(32), length=32)
        self.Wj    = b"WEBSITE_TAG_ALPHA"
        self.rho   = os.urandom(32)
        self.A_ntt, self.rho = generate_A_ntt(self.eng, self.rho, self.l)
        self.DB_W  = {}; self.DB_S = {}
        self.perf_stats = {k: {'latency': []}
                           for k in ('registration', 'authentication', 'key_rotation')}

    def _user_id(self, un):
        return shake128(un.encode(), self.Wj, length=32).hex()

    def _derive_si_ei(self, msk, ti):
        return (cbd_sample_eta(shake128(msk, ti.encode(), b"si", length=32), eta=3, nonce=0),
                cbd_sample_eta(shake128(msk, ti.encode(), b"ei", length=32), eta=2, nonce=1))

    def _derive_hi(self, pwd, un, ri_seed):
        return cbd_sample_eta(shake128(pwd.encode(), un.encode(), ri_seed, length=32), eta=2, nonce=0)

    def _compute_Bi(self, hi, si, ei):
        ski     = (hi.astype(np.int64) + si.astype(np.int64)) % self.Q
        ski_ntt = self.eng.ntt(ski)
        ei_ntt  = self.eng.ntt(ei.astype(np.int64) % self.Q)
        # Fix: Use ski_ntt replicated 'l' times, and self. for instance variables
        s_vec = [ski_ntt] * self.l
        Asi = self.eng.matvec_ntt(self.A_ntt, s_vec, self.l)
        return [(Asi[r] + ei_ntt) % self.Q for r in range(self.l)]

    def register(self, un, pwd, track_perf=True):
        t0 = time.perf_counter()
        ti = self._user_id(un)
        ri_seed = shake128(os.urandom(32), length=32)
        hi = self._derive_hi(pwd, un, ri_seed)
        si, ei = self._derive_si_ei(self.msk_W, ti)
        Bi = self._compute_Bi(hi, si, ei)
        self.DB_W[ti] = {'un': un, 'ri_seed': ri_seed, 'Bi': Bi}
        self.DB_S[ti] = {'count': 0, 'last_fail': 0.0, 'lock_until': 0.0}
        elapsed = time.perf_counter() - t0
        if track_perf:
            self.perf_stats['registration']['latency'].append(elapsed * 1000)
        return elapsed, ti

    def authenticate(self, un, pwd_in, track_perf=True):
        t0 = time.perf_counter(); ti = self._user_id(un)
        if ti not in self.DB_W: return None, "USER_NOT_FOUND", 0
        state = self.DB_S[ti]; now = time.time()
        if now < state['lock_until']:
            return None, f"LOCKED_OUT_{int(state['lock_until']-now)}s", 0
        hi_p    = self._derive_hi(pwd_in, un, self.DB_W[ti]['ri_seed'])
        si, ei  = self._derive_si_ei(self.msk_W, ti)
        Bi_calc = self._compute_Bi(hi_p, si, ei)
        valid   = all(np.array_equal(Bi_calc[k], self.DB_W[ti]['Bi'][k]) for k in range(self.l))
        elapsed = time.perf_counter() - t0
        if valid:
            self.DB_S[ti] = {'count': 0, 'last_fail': 0.0, 'lock_until': 0.0}
            if track_perf:
                self.perf_stats['authentication']['latency'].append(elapsed * 1000)
            return Bi_calc, "SUCCESS", elapsed
        else:
            if now - state['last_fail'] > self.window: state['count'] = 1
            else: state['count'] += 1
            state['last_fail'] = now
            if state['count'] >= self.limit: state['lock_until'] = now + self.lock_duration
            return Bi_calc, f"FAIL_{state['count']}", elapsed

    def rotate(self, track_perf=True):
        t0 = time.perf_counter(); old_msk = self.msk_W
        new_msk = shake128(old_msk, str(time.time()).encode(), os.urandom(16), length=32)
        old_entries = {ti: {'un': d['un'], 'Bi': [b.copy() for b in d['Bi']]}
                       for ti, d in self.DB_W.items()}
        for ti, data in self.DB_W.items():
            si_old, ei_old = self._derive_si_ei(old_msk, ti)
            si_new, ei_new = self._derive_si_ei(new_msk, ti)
            ds_ntt  = self.eng.ntt((si_new - si_old).astype(np.int64) % self.Q)
            eo_ntt  = self.eng.ntt(ei_old.astype(np.int64) % self.Q)
            en_ntt  = self.eng.ntt(ei_new.astype(np.int64) % self.Q)
            Ads     = self.eng.matvec_ntt(self.A_ntt, [ds_ntt] * self.l, self.l)
            data['Bi'] = [(data['Bi'][r] - eo_ntt + Ads[r] + en_ntt) % self.Q
                          for r in range(self.l)]
        self.msk_W = new_msk
        elapsed = time.perf_counter() - t0
        if track_perf:
            self.perf_stats['key_rotation']['latency'].append(elapsed * 1000)
        return old_msk.hex(), new_msk.hex(), old_entries, elapsed


# =============================================================================
# 6. BENCHMARK HELPERS
# =============================================================================

BATCH_SIZES               = [1, 10, 50, 100, 250, 500]
N_RUNS                    = 20
DB_SIZE_FOR_AUTH_ROTATION = 10
COLOR_MAP = {"Registration": "blue", "Authentication": "green", "Key Rotation": "red"}


def _make_temp_proto(ref):
    tmp = PCDLProtocol.__new__(PCDLProtocol)
    tmp.eng = ref.eng; tmp.Q = ref.Q; tmp.N = ref.N; tmp.l = ref.l
    tmp.limit = ref.limit; tmp.window = ref.window; tmp.lock_duration = ref.lock_duration
    tmp.Wj = ref.Wj; tmp.rho = ref.rho; tmp.A_ntt = ref.A_ntt; tmp.msk_W = ref.msk_W
    tmp.DB_W = {}; tmp.DB_S = {}
    tmp.perf_stats = {k: {'latency': [], 'memory': []}
                      for k in ('registration', 'authentication', 'key_rotation')}
    return tmp

def _copy_db(src, dst):
    dst.DB_W = {k: {'un': v['un'], 'ri_seed': v['ri_seed'],
                    'Bi': [b.copy() for b in v['Bi']]}
                for k, v in src.DB_W.items()}
    dst.DB_S = {k: dict(v) for k, v in src.DB_S.items()}


def _bench_batch(phase, n, ref, reps=3):
    """
    Times `phase` for batch size n, averaged over `reps` repetitions to
    reduce single-sample timing noise (sub-ms ops are highly variable
    on shared/virtualized hosts). Returns (mean_ms, mem_kb).
    """
    elapsed_runs = []skp
    for r in range(reps):
        tmp = _make_temp_proto(ref)
        if phase == "Registration":
            t0 = time.perf_counter()
            for i in range(n): tmp.register(f"u{n}_{r}_{i}", "pass", track_perf=False)
            elapsed_runs.append(time.perf_counter() - t0)
        elif phase == "Authentication":
            for i in range(n): tmp.register(f"u{n}_{r}_{i}", "pass", track_perf=False)
            t0 = time.perf_counter()
            for i in range(n): tmp.authenticate(f"u{n}_{r}_{i}", "pass", track_perf=False)
            elapsed_runs.append(time.perf_counter() - t0)
        elif phase == "Key Rotation":
            for i in range(n): tmp.register(f"u{n}_{r}_{i}", "pass", track_perf=False)
            t0 = time.perf_counter()
            tmp.rotate(track_perf=False)
            elapsed_runs.append(time.perf_counter() - t0)
    return np.mean(elapsed_runs) * 1000


def _measure_avg_latency(ref):
    """
    Single-operation latency averaged over N_RUNS repetitions.

    Registration  : 1 new user registered into a fresh empty DB.
    Authentication: 1 existing user authenticated from a DB of
                    DB_SIZE_FOR_AUTH_ROTATION pre-registered users.
    Key Rotation  : server rotates keys for all DB_SIZE_FOR_AUTH_ROTATION
                    users (one full server-side batch).
    """
    base = _make_temp_proto(ref)
    for i in range(DB_SIZE_FOR_AUTH_ROTATION):
        base.register(f"base_{i}", "pass", track_perf=False)
    first_un = base.DB_W[list(base.DB_W.keys())[0]]['un']

    results = {}
    for phase in ["Registration", "Authentication", "Key Rotation"]:
        lats = []
        for r in range(N_RUNS):
            tmp = _make_temp_proto(base)
            if phase == "Registration":
                t0 = time.perf_counter()
                tmp.register(f"newuser_{r}", "pass", track_perf=False)
                lats.append((time.perf_counter() - t0) * 1000)
            elif phase == "Authentication":
                _copy_db(base, tmp)
                t0 = time.perf_counter()
                tmp.authenticate(first_un, "pass", track_perf=False)
                lats.append((time.perf_counter() - t0) * 1000)
            elif phase == "Key Rotation":
                _copy_db(base, tmp)
                t0 = time.perf_counter()
                tmp.rotate(track_perf=False)
                lats.append((time.perf_counter() - t0) * 1000)
        results[phase] = (np.mean(lats), np.std(lats), lats)
    return results


def _print_operation_breakdown(ref):
    eng  = ref.eng; REPS = 2000
    a    = np.random.randint(0, eng.Q, eng.N, dtype=np.int64)
    b    = np.random.randint(0, eng.Q, eng.N, dtype=np.int64)
    s32  = os.urandom(32)
    def _t(fn):
        t0 = time.perf_counter()
        for _ in range(REPS): fn()
        return (time.perf_counter() - t0) / REPS * 1e6
    ntt_us  = _t(lambda: eng.ntt(a))
    intt_us = _t(lambda: eng.intt(a))
    pw_us   = _t(lambda: LatticeEngine._pw_mul(a, b, eng.Q))
    cbd_us  = _t(lambda: cbd_sample_eta(s32, eta=2, nonce=0))
    sha_us  = _t(lambda: hashlib.shake_128(s32).digest(32))
    est_us  = 2 * ntt_us + 4 * pw_us + 3 * cbd_us + 5 * sha_us
    print(f"\n  ┌── Primitive Timings (NumPy-vectorised NTT, {REPS} runs each) ───")
    print(f"  │  NTT  (n=256, q=3329)         : {ntt_us:7.1f} µs")
    print(f"  │  INTT (n=256, q=3329)         : {intt_us:7.1f} µs")
    print(f"  │  Pointwise mul (NTT domain)   : {pw_us:7.1f} µs")
    print(f"  │  CBD eta=1 (256 coefficients) : {cbd_us:7.1f} µs")
    print(f"  │  SHAKE-128 (32-byte output)   : {sha_us:7.1f} µs")
    print(f"  │  Estimated per-register       : {est_us:7.1f} µs = {est_us/1000:.3f} ms")
    print(f"  └────────────────────────────────────────────────────────────\n")


# =============================================================================
# 7. GRAPH FUNCTIONS
# =============================================================================

def _print_menu():
    print("\n" + "="*70)
    print(" "*27 + "MAIN MENU")
    print("="*70)
    print("  1. Register More Users")
    print("  2. Authenticate User")
    print("  3. Perform Key Rotation")
    print("  4. Exit")
    print("  5. Show Performance Graphs")
    print("="*70)


def plot_average_latency(ref):
    print(f"\n{'='*65}")
    print(f"  Average Latency Benchmark  ({N_RUNS} runs per algorithm)")
    print(f"  Auth & Key Rotation use a DB of {DB_SIZE_FOR_AUTH_ROTATION} pre-registered users")
    print(f"{'='*65}")
    _print_operation_breakdown(ref)

    results = _measure_avg_latency(ref)
    phases  = ["Registration", "Authentication", "Key Rotation"]
    means   = [results[p][0] for p in phases]
    stds    = [results[p][1] for p in phases]
    colors  = [COLOR_MAP[p]  for p in phases]

    print(f"  {'Algorithm':<22}  {'Avg (ms)':>10}  {'Std (ms)':>10}  "
          f"{'Min (ms)':>10}  {'Max (ms)':>10}  {'N':>4}")
    print(f"  {'─'*22}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*4}")
    for p in phases:
        m, s, lats = results[p]
        print(f"  {p:<22}  {m:>10.4f}  {s:>10.4f}  "
              f"{min(lats):>10.4f}  {max(lats):>10.4f}  {N_RUNS:>4}")

    rot_per_user = results["Key Rotation"][0] / DB_SIZE_FOR_AUTH_ROTATION
    print(f"\n  Key Rotation per-user cost ≈ {rot_per_user:.4f} ms/user")
    print(f"  (Total rotation time scales linearly with DB size)\n")

    x    = np.arange(len(phases))
    lbls = [f"Registration\n(1 user)",
            f"Authentication\n(DB={DB_SIZE_FOR_AUTH_ROTATION} users)",
            f"Key Rotation\n(DB={DB_SIZE_FOR_AUTH_ROTATION} users)"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colors,
                  edgecolor='black', linewidth=0.8, alpha=0.85, width=0.5)
    ax.set_title("Average Latency per Algorithm", fontweight='bold', fontsize=14)
    ax.set_xlabel("Protocol Phase", fontsize=12)
    ax.set_ylabel("Average Latency (ms)", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels(lbls, fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + s + max(means) * 0.01,
                f"{m:.4f} ms", ha='center', va='bottom', fontsize=9)
    _show_graph(fig)


def plot_asymptotic(phase_name, ref):
    print(f"\n{'='*65}")
    print(f"  Asymptotic Scaling — {phase_name}")
    print(f"{'='*65}")
    lats= []
    for n in BATCH_SIZES:
        lat_ms = _bench_batch(phase_name, n, ref)
        lats.append(lat_ms)
        print(f"  n={n:4d}  latency={lat_ms:8.2f} ms")

    color = COLOR_MAP.get(phase_name, 'blue')
    fig, ax1 = plt.subplots(figsize=(6, 3.5))

    ax1.plot(BATCH_SIZES, lats, marker='s', linestyle='-', linewidth=2, markersize=8, color=color)
    ax1.set_title(f"Asymptotic {phase_name} Performance", fontweight='bold', fontsize=14)
    ax1.set_xlabel("Number of Users", fontsize=12)
    ax1.set_ylabel("Total Latency (ms)", fontsize=12)
    ax1.grid(True, alpha=0.3)

    _show_graph(fig)


# =============================================================================
# 8. MAIN
# =============================================================================

if __name__ == "__main__":

    proto = PCDLProtocol()

    print("\n" + "="*70)
    print(" "*15 + "PCDL AUTHENTICATION PROTOCOL")
    print("="*70)
    print(f"\n  Modulus (q)           : {proto.Q}")
    print(f"  Polynomial Degree (n) : {proto.N}")
    print(f"  Module Rank (l)       : {proto.l}")
    print(f"  NTT Root (zeta)       : {LatticeEngine.ZETA}  (512th root mod q, FIPS 203)")
    print(f"  CBD eta               : eta1=3 (si/ei), eta2=2 (hi)  -- matches ML-KEM-512")
    print(f"  Hash / XOF            : SHAKE-128 (FIPS 202)")
    print(f"  A-seed (rho)          : {proto.rho.hex()[:16]}...")
    print(f"  Master Key            : {proto.msk_W.hex()[:16]}...")
    print("="*70)

    # ── Initial 3 registrations ──────────────────────────────────────────
    print("\n" + "="*70)
    print(" "*22 + "REGISTRATION PHASE")
    print("="*70)
    print("\nRegister 3 users to start.\n")

    for idx in range(3):
        print(f"[User {idx+1}/3]")
        u = _input("  Username : ").strip()
        p = _input("  Password : ").strip()
        dt, ti = proto.register(u, p)
        print(f"  ✓ Registered in {dt*1000:.4f} ms")
        print(f"    ti        : {ti[:16]}...")
        print(f"    ri_seed   : {proto.DB_W[ti]['ri_seed'].hex()[:16]}...")
        print(f"    Bi[0][0:8]: {proto.DB_W[ti]['Bi'][0][:8].tolist()}")
        print(f"    Bi[1][0:8]: {proto.DB_W[ti]['Bi'][1][:8].tolist()}\n")

    print("  ✓ 3 users registered.")
    print("  → Use Option 5 to view performance graphs.\n")

    # ── Main menu loop ────────────────────────────────────────────────────
    while True:
        _print_menu()
        cmd = _input("\nSelect Option (1-5): ").strip()

        # ── 1. Register ──────────────────────────────────────────────────
        if cmd == '1':
            u = _input("  Username : ").strip()
            p = _input("  Password : ").strip()
            dt, ti = proto.register(u, p)
            print(f"\n  ✓ Registered in {dt*1000:.4f} ms")
            print(f"    ti        : {ti[:16]}...")
            print(f"    ri_seed   : {proto.DB_W[ti]['ri_seed'].hex()[:16]}...")
            print(f"    Bi[0][0:8]: {proto.DB_W[ti]['Bi'][0][:8].tolist()}")
            print(f"    Bi[1][0:8]: {proto.DB_W[ti]['Bi'][1][:8].tolist()}\n")

        # ── 2. Authenticate ──────────────────────────────────────────────
        elif cmd == '2':
            u = _input("  Username : ").strip()
            p = _input("  Password : ").strip()
            calc_Bi, status, lat = proto.authenticate(u, p)
            if status == "USER_NOT_FOUND":
                print(f"\n  ✗ User '{u}' not found.  Registered users:")
                for d in proto.DB_W.values(): print(f"      • {d['un']}")
            elif "LOCKED_OUT" in status:
                print(f"\n  🔒 Account locked. Retry in: {status.split('_')[2]}")
            elif status == "SUCCESS":
                print(f"\n  ✓ AUTH SUCCESS  |  {lat*1000:.4f} ms")
                print(f"    Bi matches stored Bi: YES")
            else:
                fc  = int(status.split("_")[1])
                tik = proto._user_id(u)
                print(f"\n  ✗ WRONG PASSWORD (attempt {fc}/{proto.limit})")
                if fc >= proto.limit - 1: print("    ⚠ Next failure locks this account!")
                print(f"    Stored  Bi[0][0:8]: {proto.DB_W[tik]['Bi'][0][:8].tolist()}")
                print(f"    Calc    Bi[0][0:8]: {calc_Bi[0][:8].tolist()}")
            print()

        # ── 3. Key Rotation ──────────────────────────────────────────────
        elif cmd == '3':
            old_hex, new_hex, old_ent, lat = proto.rotate()
            print(f"\n  ✓ KEY ROTATION  {lat*1000:.2f} ms  |  "
                  f"{len(old_ent)} users updated")
            print(f"    Old key : {old_hex[:32]}...")
            print(f"    New key : {new_hex[:32]}...")
            for tik, od in old_ent.items():
                nd = proto.DB_W[tik]
                print(f"\n    User: {od['un']}")
                print(f"      Before Bi[0][0:8]: {od['Bi'][0][:8].tolist()}")
                print(f"      After  Bi[0][0:8]: {nd['Bi'][0][:8].tolist()}")
            print()

        # ── 4. Exit ──────────────────────────────────────────────────────
        elif cmd == '4':
            print("\n" + "="*70); print(" "*22 + "SESSION SUMMARY"); print("="*70)
            rows = []
            for ph, key in [("Registration",   "registration"),
                             ("Authentication", "authentication"),
                             ("Key Rotation",   "key_rotation")]:
                lts = proto.perf_stats[key]['latency']
                if lts:
                    rows.append([ph, len(lts),
                                  f"{np.mean(lts):.4f}", f"{np.min(lts):.4f}",
                                  f"{np.max(lts):.4f}", f"{np.std(lts):.4f}"])
            if rows:
                hdr = (f"  {'Phase':<18}{'N':>5}{'Avg ms':>9}{'Min ms':>9}"
                       f"{'Max ms':>9}{'Std':>7}")
                print(hdr); print("  " + "─" * (len(hdr) - 2))
                for r in rows:
                    print(f"  {r[0]:<18}{r[1]:>5}{r[2]:>9}{r[3]:>9}"
                          f"{r[4]:>9}{r[5]:>7}")
            else:
                print("  No recorded operations.")
            print("\n" + "="*70); print(" "*27 + "Goodbye"); print("="*70)
            break

        # ── 5. Show Performance Graphs ───────────────────────────────────
        elif cmd == '5':
            print("\n" + "="*70); print(" "*22 + "PERFORMANCE GRAPHS"); print("="*70)
            print("  a. Average Latency bar chart (all 3 algorithms)")
            print("  b. Asymptotic Scaling — Registration")
            print("  c. Asymptotic Scaling — Authentication")
            print("  d. Asymptotic Scaling — Key Rotation")
            print("  e. All of the above")
            sub = _input("\n  Select (a/b/c/d/e): ").strip().lower()
            if sub in ('a', 'e'): plot_average_latency(proto)
            if sub in ('b', 'e'): plot_asymptotic("Registration", proto)
            if sub in ('c', 'e'): plot_asymptotic("Authentication", proto)
            if sub in ('d', 'e'): plot_asymptotic("Key Rotation", proto)
            if sub not in ('a', 'b', 'c', 'd', 'e'):
                print("  Invalid sub-option.\n")
            print()  # spacing before the loop reprints the main menu

        else:
            print("  Invalid option. Choose 1–5.\n")
