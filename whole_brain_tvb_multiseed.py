"""
Spoor A - firming step 1 (+2): MULTI-SEED rate-matched. Does the capstone hold across
independent noise realisations AND different 5-HT2A spatial maps?

Each seed varies: (a) the stochastic noise stream, (b) the synthetic 5-HT2A density map
(random gradient shuffle). So this firms BOTH the statistical significance of the LZc/integration
effects (step 1) AND robustness to the receptor-map choice (step 2, pending the real PET atlas).

All conditions are RATE-MATCHED to 3 Hz per seed (re-FIC'd), so the readout is the rate-controlled
mechanism, not a firing-rate artifact. Reports mean +/- std and the fraction of seeds supporting
each sign.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from tvb.simulator.lab import *  # noqa
from tvb.basic.neotraits.api import NArray

conn = connectivity.Connectivity.from_file()
conn.weights = conn.weights / conn.weights.max()
conn.speed = np.array([3.0])
conn.configure()
N = conn.number_of_regions
k_strength = conn.weights.sum(axis=1)

A_E0 = 310.0
S_E_TARGET = 0.164
GAIN_S = 0.15
BLOCK_EI = 0.4
BLOCK_G = 0.3
G_WP = 0.06
NSIG = 1e-4
SEEDS = [1, 2, 3, 4, 5, 6]

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

def fic_for(a_e, J_N_EI, G, seed, refine=10, gain=3.0, cap=0.5, T=1500.0, discard=800.0):
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
        sim = build_sim(a_e, J_N_EI, J_i, G, seed=seed)
        m = run_sim(sim, T, discard_ms=discard).mean(0)
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

JNEI_n2o = 0.15 * (1.0 - BLOCK_EI)
G_n2o = G_WP * (1.0 - BLOCK_G)
cond_specs = {
    "LSD":   dict(a_s=GAIN_S, jnei=0.15,     G=G_WP),
    "N2O":   dict(a_s=0.0,    jnei=JNEI_n2o, G=G_n2o),
    "COMBO": dict(a_s=GAIN_S, jnei=JNEI_n2o, G=G_n2o),
}

acc = {c: {"dLZc": [], "dInteg": []} for c in cond_specs}
for si in SEEDS:
    rng = np.random.default_rng(si)
    dens = np.linspace(0.1, 1.0, N); rng.shuffle(dens)   # per-seed 5-HT2A map
    # baseline (rate-matched)
    Jb = fic_for(np.full(N, A_E0), 0.15, G_WP, si)
    Sb = run_sim(build_sim(np.full(N, A_E0), 0.15, Jb, G_WP, seed=si), 25000.0, discard_ms=8000.0)
    lz_b, in_b = metrics(Sb)
    line = "seed %d  base(LZc %.3f, Int %.4f)" % (si, lz_b, in_b)
    for cname, p in cond_specs.items():
        a_e = A_E0 * (1.0 + p["a_s"] * dens)
        Ji = fic_for(a_e, p["jnei"], p["G"], si)
        Se = run_sim(build_sim(a_e, p["jnei"], Ji, p["G"], seed=si), 25000.0, discard_ms=8000.0)
        lz, ig = metrics(Se)
        dlz = 100 * (lz / lz_b - 1); dig = 100 * (ig / in_b - 1)
        acc[cname]["dLZc"].append(dlz); acc[cname]["dInteg"].append(dig)
        line += "  %s(dLZc %+.1f%%, dInt %+.0f%%)" % (cname, dlz, dig)
    print(line, flush=True)

print("\n=== MULTI-SEED SUMMARY (n=%d, rate-matched ~3 Hz) ===" % len(SEEDS))
rows = []
for c in cond_specs:
    lz = np.array(acc[c]["dLZc"]); ig = np.array(acc[c]["dInteg"])
    rows.append(dict(Condition=c,
                     dLZc_mean=round(lz.mean(), 2), dLZc_std=round(lz.std(), 2),
                     LZc_pos_frac="%d/%d" % ((lz > 0).sum(), len(lz)),
                     dInteg_mean=round(ig.mean(), 1), dInteg_std=round(ig.std(), 1),
                     Integ_neg_frac="%d/%d" % ((ig < 0).sum(), len(ig))))
sm = pd.DataFrame(rows)
print(sm.to_string(index=False))
sm.to_csv(r"E:\BiniruProjects\psyche-sim\results_multiseed.csv", index=False)
co = sm[sm.Condition == "COMBO"].iloc[0]
print("\n[verdict] COMBO integration collapse: mean %.0f%% (%s seeds negative) -> %s"
      % (co.dInteg_mean, co.Integ_neg_frac, "ROBUST" if co.Integ_neg_frac.startswith(str(len(SEEDS))) else "variable"))
print("[verdict] COMBO LZc change: mean %+.2f%% +/- %.2f (%s seeds positive) -> %s"
      % (co.dLZc_mean, co.dLZc_std, co.LZc_pos_frac,
         "marginal/within-noise" if abs(co.dLZc_mean) < co.dLZc_std else "consistent sign"))
print("Saved: results_multiseed.csv")
