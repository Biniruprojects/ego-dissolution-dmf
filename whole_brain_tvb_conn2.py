"""
Spoor A - firming step 4b: SECOND CONNECTOME. Does the integration collapse generalise to a
different brain topology? Tries an alternative empirical tvb-data connectome (96/192 regions);
falls back to a structurally distinct synthetic small-world if none is available. Runs the
rate-matched COMBO (+ baseline) over 2 seeds and reports the collapse + LZc change.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import networkx as nx
import pandas as pd
from tvb.simulator.lab import *  # noqa
from tvb.basic.neotraits.api import NArray

# --- pick a SECOND connectome, different from the default 76-region one ---
conn = None
for nm in ["connectivity_192.zip", "connectivity_96.zip", "connectivity_66.zip"]:
    try:
        c = connectivity.Connectivity.from_file(nm)
        c.weights = c.weights / c.weights.max()
        c.speed = np.array([3.0])
        c.configure()
        conn = c
        print("[conn2] loaded empirical %s -> %d regions" % (nm, c.number_of_regions), flush=True)
        break
    except Exception as e:
        print("[conn2] %s not available (%s)" % (nm, type(e).__name__), flush=True)
if conn is None:
    Ns = 100
    g = nx.watts_strogatz_graph(Ns, k=10, p=0.5, seed=42)   # different topology than before
    W = nx.to_numpy_array(g) * np.random.default_rng(42).uniform(0.5, 1.5, (Ns, Ns))
    W = (W + W.T) / 2; np.fill_diagonal(W, 0.0); W /= W.max()
    conn = connectivity.Connectivity(weights=W, tract_lengths=np.zeros((Ns, Ns)),
                                     region_labels=np.array(["R%03d" % i for i in range(Ns)]),
                                     centres=np.zeros((Ns, 3)))
    conn.speed = np.array([np.inf]); conn.configure()
    print("[conn2] using synthetic %d-region small-world (k=10,p=0.5)" % Ns, flush=True)

N = conn.number_of_regions
k_strength = conn.weights.sum(axis=1)

A_E0 = 310.0; S_E_TARGET = 0.164; G_WP = 0.06; NSIG = 1e-4
GAIN_S, BLOCK_EI, BLOCK_G = 0.15, 0.4, 0.3

class N2OWongWang(models.ReducedWongWangExcInh):
    J_N_EI = NArray(default=np.array([0.15]), doc="E->I NMDA conductance")
    def dfun(self, x, c, local_coupling=0.0, **kwargs):
        S = x
        ae = np.reshape(self.a_e, (-1, 1)); ji = np.reshape(self.J_i, (-1, 1))
        coup = self.G * self.J_N * (c[0] + local_coupling * S[0])
        JNSe = self.J_N * S[0]
        x_e = self.w_p * JNSe - ji * S[1] + self.W_e * self.I_o + coup + self.I_ext
        x_e = ae * x_e - self.b_e
        H_e = x_e / (1.0 - np.exp(-self.d_e * x_e))
        dS_e = -(S[0] / self.tau_e) + (1.0 - S[0]) * H_e * self.gamma_e
        x_i = self.J_N_EI * S[0] - S[1] + self.W_i * self.I_o + self.lamda * coup
        x_i = self.a_i * x_i - self.b_i
        H_i = x_i / (1.0 - np.exp(-self.d_i * x_i))
        dS_i = -(S[1] / self.tau_i) + H_i * self.gamma_i
        return np.array([dS_e, dS_i])

def s_e_to_rate(s):
    return s / (100.0 * 0.000641 * (1.0 - s))

def build_sim(a_e, J_N_EI, J_i, G, seed=0, dt=1.0):
    m = N2OWongWang()
    m.G = np.array([float(G)]); m.J_N = np.array([0.15]); m.J_N_EI = np.array([float(J_N_EI)])
    m.a_e = np.asarray(a_e, float).reshape(N); m.J_i = np.asarray(J_i, float).reshape(N)
    m.variables_of_interest = ['S_e']
    ic = np.zeros((200, 2, N, 1)); ic[:, 0, :, 0] = 0.164; ic[:, 1, :, 0] = 0.04
    nz = noise.Additive(nsig=np.array([NSIG]))
    try:
        nz.random_stream = np.random.RandomState(seed)
    except Exception:
        pass
    sim = simulator.Simulator(
        model=m, connectivity=conn, coupling=coupling.Linear(a=np.array([1.0])),
        integrator=integrators.HeunStochastic(dt=dt, noise=nz),
        monitors=[monitors.TemporalAverage(period=10.0)], initial_conditions=ic)
    sim.configure()
    return sim

def run_sim(sim, T, discard_ms=0.0):
    t, d = sim.run(simulation_length=T)[0]
    S_e = d[:, 0, :, 0]
    return S_e[int(discard_ms / 10.0):] if discard_ms > 0 else S_e

def H_I_scalar(x):
    y = 615.0 * x - 177.0
    return y / (1.0 - np.exp(min(-0.087 * y, 50.0)))

def fic_for(a_e, J_N_EI, G, seed, refine=15, gain=3.0, cap=0.5, T=1500.0, discard=800.0):
    a_e = np.asarray(a_e, float).reshape(N)
    gI, tI = 0.001, 10.0
    wp, JN, We, Wi, Io = 1.4, 0.15, 1.0, 0.7, 0.382
    S_E = 0.164
    def g(S_I):
        return tI * gI * H_I_scalar(Wi * Io + J_N_EI * S_E - S_I) - S_I
    lo, hi = 1e-4, 0.20; glo = g(lo)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if glo * g(mid) <= 0: hi = mid
        else: lo = mid; glo = g(lo)
    S_I = 0.5 * (lo + hi)
    I_E_target = (125.0 - 8.0) / a_e
    J_i = np.clip((We * Io + wp * JN * S_E + G * JN * S_E * k_strength - I_E_target) / S_I, 0.001, None)
    for _ in range(refine):
        m = run_sim(build_sim(a_e, J_N_EI, J_i, G, seed=seed), T, discard_ms=discard).mean(0)
        err = m - S_E_TARGET
        if np.max(np.abs(err)) < 0.012: break
        J_i = np.clip(J_i + np.clip(gain * err, -cap, cap), 0.001, None)
    return J_i

def lz76(b):
    s = ''.join('1' if v else '0' for v in b); n = len(s)
    if n < 2: return 0.0
    i, c, l = 0, 1, 1; k, kmax = 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n: c += 1; break
        else:
            kmax = max(kmax, k); i += 1
            if i == l:
                c += 1; l += kmax
                if l + 1 > n: break
                i = 0; k = 1; kmax = 1
            else: k = 1
    return c / (n / np.log2(n))

def metrics(S_e):
    Sc = S_e[::5]
    lzc = np.mean([lz76(Sc[:, j] > np.median(Sc[:, j])) for j in range(N)])
    Sz = (S_e - S_e.mean(0)) / (S_e.std(0) + 1e-9)
    FC = np.corrcoef(Sz.T); iu = np.triu_indices(N, 1)
    return lzc, FC[iu].mean()

JNEI = 0.15 * (1.0 - BLOCK_EI); Gn = G_WP * (1.0 - BLOCK_G)
res = []
for si in [1, 2]:
    rng = np.random.default_rng(si)
    dens = np.linspace(0.1, 1.0, N); rng.shuffle(dens)
    Jb = fic_for(np.full(N, A_E0), 0.15, G_WP, si)
    lz_b, in_b = metrics(run_sim(build_sim(np.full(N, A_E0), 0.15, Jb, G_WP, seed=si), 25000.0, discard_ms=8000.0))
    a_e = A_E0 * (1.0 + GAIN_S * dens)
    Jc = fic_for(a_e, JNEI, Gn, si)
    lz_c, in_c = metrics(run_sim(build_sim(a_e, JNEI, Jc, Gn, seed=si), 25000.0, discard_ms=8000.0))
    dlz = 100 * (lz_c / lz_b - 1); dig = 100 * (in_c / in_b - 1)
    res.append((dlz, dig))
    print("  seed %d: COMBO dLZc=%+.1f%%  dInteg=%+.0f%%" % (si, dlz, dig), flush=True)

dlz = np.array([r[0] for r in res]); dig = np.array([r[1] for r in res])
print("\n=== SECOND CONNECTOME (%d regions) ===" % N)
print("COMBO dLZc = %+.2f%% (mean)   dInteg = %+.0f%% (mean)" % (dlz.mean(), dig.mean()))
print("[verdict] integration collapse generalises to a different topology: %s"
      % ("YES" if dig.mean() < -10 else "weaker/no"))
pd.DataFrame(dict(seed=[1, 2], dLZc=dlz, dInteg=dig)).to_csv(
    r"E:\BiniruProjects\psyche-sim\results_conn2.csv", index=False)
print("Saved: results_conn2.csv")
