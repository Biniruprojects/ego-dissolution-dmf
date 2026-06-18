"""
Resolve the apparent real-map vs synthetic-map discrepancy: run the EMPIRICAL Beliveau Cimbi-36
5-HT2A map through the SAME 6-seed multi-seed protocol (fixed real map, noise varies per seed,
all conditions rate-matched to 3 Hz). This yields a mean +/- std directly comparable to the
synthetic multi-seed (combo -56% +/- 12). The single-run real-map value (-75%) was one noise
realisation; this gives the proper estimate.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from tvb.simulator.lab import *  # noqa
from tvb.basic.neotraits.api import NArray

conn = connectivity.Connectivity.from_file()
conn.weights = conn.weights / conn.weights.max()
conn.speed = np.array([3.0]); conn.configure()
N = conn.number_of_regions
k_strength = conn.weights.sum(axis=1)
DENS = np.load(r"E:\BiniruProjects\psyche-sim\ht2a_76.npy")   # the REAL map, fixed across seeds

A_E0 = 310.0; S_E_TARGET = 0.164; GAIN_S = 0.15; BLOCK_EI = 0.4; BLOCK_G = 0.3; G_WP = 0.06; NSIG = 1e-4
SEEDS = [1, 2, 3, 4, 5, 6]

class N2OWongWang(models.ReducedWongWangExcInh):
    J_N_EI = NArray(default=np.array([0.15]))
    def dfun(self, x, c, local_coupling=0.0, **kwargs):
        S = x; ae = np.reshape(self.a_e, (-1, 1)); ji = np.reshape(self.J_i, (-1, 1))
        coup = self.G * self.J_N * (c[0] + local_coupling * S[0]); JNSe = self.J_N * S[0]
        x_e = self.w_p * JNSe - ji * S[1] + self.W_e * self.I_o + coup + self.I_ext
        x_e = ae * x_e - self.b_e; H_e = x_e / (1.0 - np.exp(-self.d_e * x_e))
        dS_e = -(S[0] / self.tau_e) + (1.0 - S[0]) * H_e * self.gamma_e
        x_i = self.J_N_EI * S[0] - S[1] + self.W_i * self.I_o + self.lamda * coup
        x_i = self.a_i * x_i - self.b_i; H_i = x_i / (1.0 - np.exp(-self.d_i * x_i))
        dS_i = -(S[1] / self.tau_i) + H_i * self.gamma_i
        return np.array([dS_e, dS_i])

def s_e_to_rate(s):
    return s / (100.0 * 0.000641 * (1.0 - s))

def build(a_e, jnei, J_i, G, seed):
    m = N2OWongWang(); m.G = np.array([float(G)]); m.J_N = np.array([0.15]); m.J_N_EI = np.array([float(jnei)])
    m.a_e = np.asarray(a_e, float).reshape(N); m.J_i = np.asarray(J_i, float).reshape(N)
    m.variables_of_interest = ['S_e']
    ic = np.zeros((200, 2, N, 1)); ic[:, 0, :, 0] = 0.164; ic[:, 1, :, 0] = 0.04
    nz = noise.Additive(nsig=np.array([NSIG]))
    try:
        nz.random_stream = np.random.RandomState(seed)
    except Exception:
        pass
    sim = simulator.Simulator(model=m, connectivity=conn, coupling=coupling.Linear(a=np.array([1.0])),
        integrator=integrators.HeunStochastic(dt=1.0, noise=nz),
        monitors=[monitors.TemporalAverage(period=10.0)], initial_conditions=ic)
    sim.configure(); return sim

def run(sim, T, disc):
    d = sim.run(simulation_length=T)[0][1][:, 0, :, 0]
    return d[int(disc / 10.0):]

def H_I_scalar(x):
    y = 615.0 * x - 177.0
    return y / (1.0 - np.exp(min(-0.087 * y, 50.0)))

def fic(a_e, jnei, G, seed, refine=12):
    a_e = np.asarray(a_e, float).reshape(N)
    gI, tI = 0.001, 10.0; wp, JN, We, Wi, Io = 1.4, 0.15, 1.0, 0.7, 0.382; S_E = 0.164
    def g(S_I): return tI * gI * H_I_scalar(Wi * Io + jnei * S_E - S_I) - S_I
    lo, hi = 1e-4, 0.20; glo = g(lo)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if glo * g(mid) <= 0: hi = mid
        else: lo = mid; glo = g(lo)
    S_I = 0.5 * (lo + hi)
    J_i = np.clip((We * Io + wp * JN * S_E + G * JN * S_E * k_strength - (125.0 - 8.0) / a_e) / S_I, 0.001, None)
    for _ in range(refine):
        m = run(build(a_e, jnei, J_i, G, seed), 1500.0, 800.0).mean(0); err = m - S_E_TARGET
        if np.max(np.abs(err)) < 0.012: break
        J_i = np.clip(J_i + np.clip(3.0 * err, -0.5, 0.5), 0.001, None)
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
    Sc = S_e[::5]; lzc = np.mean([lz76(Sc[:, j] > np.median(Sc[:, j])) for j in range(N)])
    Sz = (S_e - S_e.mean(0)) / (S_e.std(0) + 1e-9); FC = np.corrcoef(Sz.T); iu = np.triu_indices(N, 1)
    return lzc, FC[iu].mean()

JNEI = 0.15 * (1 - BLOCK_EI); Gn = G_WP * (1 - BLOCK_G)
specs = {"LSD": (GAIN_S, 0.15, G_WP), "N2O": (0.0, JNEI, Gn), "COMBO": (GAIN_S, JNEI, Gn)}
acc = {c: {"dLZc": [], "dInteg": []} for c in specs}
perseed = []   # raw per-(seed,condition) rows for full traceability
for si in SEEDS:
    Jb = fic(np.full(N, A_E0), 0.15, G_WP, si)
    lz_b, in_b = metrics(run(build(np.full(N, A_E0), 0.15, Jb, G_WP, si), 25000.0, 8000.0))
    line = "seed %d base(LZc %.3f Int %.4f)" % (si, lz_b, in_b)
    for cn, (a_s, jnei, G) in specs.items():
        a_e = A_E0 * (1.0 + a_s * DENS)
        Ji = fic(a_e, jnei, G, si)
        lz, ig = metrics(run(build(a_e, jnei, Ji, G, si), 25000.0, 8000.0))
        dlz = 100 * (lz / lz_b - 1); dig = 100 * (ig / in_b - 1)
        acc[cn]["dLZc"].append(dlz); acc[cn]["dInteg"].append(dig)
        perseed.append(dict(seed=si, condition=cn, base_LZc=round(lz_b, 4), base_Integ=round(in_b, 4),
                            cond_LZc=round(lz, 4), cond_Integ=round(ig, 4),
                            dLZc_pct=round(dlz, 2), dInteg_pct=round(dig, 1)))
        line += "  %s(dLZc%+.1f dInt%+.0f)" % (cn, dlz, dig)
    print(line, flush=True)

print("\n=== REAL-MAP MULTI-SEED (n=%d, Beliveau Cimbi-36, rate-matched) ===" % len(SEEDS))
rows = []
for c in specs:
    lz = np.array(acc[c]["dLZc"]); ig = np.array(acc[c]["dInteg"])
    rows.append(dict(Condition=c, dLZc_mean=round(lz.mean(), 2), dLZc_std=round(lz.std(), 2),
                     dInteg_mean=round(ig.mean(), 1), dInteg_std=round(ig.std(), 1)))
sm = pd.DataFrame(rows); print(sm.to_string(index=False))
sm.to_csv(r"E:\BiniruProjects\psyche-sim\results_realmap_multiseed.csv", index=False)
pd.DataFrame(perseed).to_csv(r"E:\BiniruProjects\psyche-sim\results_realmap_multiseed_perseed.csv", index=False)
print("Saved per-seed rows: results_realmap_multiseed_perseed.csv (%d rows)" % len(perseed))
co = sm[sm.Condition == "COMBO"].iloc[0]
print("\n[real-map COMBO] dInteg = %.0f%% +/- %.0f   dLZc = %+.2f%% +/- %.2f" % (co.dInteg_mean, co.dInteg_std, co.dLZc_mean, co.dLZc_std))
print("(compare synthetic multi-seed: dInteg -56 +/- 12, dLZc +0.9 +/- 0.5)")
print("Saved: results_realmap_multiseed.csv")
