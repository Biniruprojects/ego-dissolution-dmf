"""
Spoor A - v8: RATE-MATCHED conditions. The confound-resolver.

Every previous result was contaminated by firing-rate entanglement: the drug perturbations
pushed mean rate from 3 Hz to 15-24 Hz, and both Lempel-Ziv complexity and FC integration
co-vary with rate. So "combo raises LZc / drops integration" might be nothing but "combo runs
the network hot". To isolate the genuine mechanism we RE-TUNE FIC per condition so EVERY
condition sits at the SAME 3 Hz baseline, then read LZc and integration at matched rate.

This is the standard control: it answers "what does each drug do to complexity / integration
AT FIXED FIRING RATE", removing the trivial rate confound. If the combo still shows high LZc +
collapsed integration at matched rate -> real. If it cancels -> the earlier 'support' was a
hot-regime artifact.

Model: faithful-N2O reduced Wong-Wang (separate E->I conductance J_N_EI), empirical connectome.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from tvb.simulator.lab import *  # noqa
from tvb.basic.neotraits.api import NArray

rng = np.random.default_rng(7)
conn = connectivity.Connectivity.from_file()
conn.weights = conn.weights / conn.weights.max()
conn.speed = np.array([3.0])
conn.configure()
N = conn.number_of_regions
k_strength = conn.weights.sum(axis=1)

dens = np.linspace(0.1, 1.0, N)
rng.shuffle(dens)

A_E0 = 310.0
S_E_TARGET = 0.164
GAIN_S = 0.15
BLOCK_EI = 0.4
BLOCK_G = 0.3
G_WP = 0.06
NSIG = 1e-4

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

def build_sim(a_e, J_N_EI, J_i, G, dt=1.0, nsig=NSIG):
    m = N2OWongWang()
    m.G = np.array([float(G)]); m.J_N = np.array([0.15]); m.J_N_EI = np.array([float(J_N_EI)])
    m.a_e = np.asarray(a_e, float).reshape(N); m.J_i = np.asarray(J_i, float).reshape(N)
    m.variables_of_interest = ['S_e']
    ic = np.zeros((200, 2, N, 1)); ic[:, 0, :, 0] = 0.164; ic[:, 1, :, 0] = 0.04
    sim = simulator.Simulator(
        model=m, connectivity=conn, coupling=coupling.Linear(a=np.array([1.0])),
        integrator=integrators.HeunStochastic(dt=dt, noise=noise.Additive(nsig=np.array([nsig]))),
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

def fic_for(a_e, J_N_EI, G, refine=30, gain=3.0, cap=0.5, T=2000.0, discard=1000.0):
    # analytic seed generalised to per-node a_e (-> per-node I_E_target) and this J_N_EI,
    # then stochastic polish. Re-tuned PER CONDITION so each sits at 3 Hz (rate-matched).
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
    I_E_target = (125.0 - 8.0) / a_e                       # per node
    J_i = np.clip((We * Io + wp * JN * S_E + G * JN * S_E * k_strength - I_E_target) / S_I, 0.001, None)
    for _ in range(refine):
        sim = build_sim(a_e, J_N_EI, J_i, G)
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
conds = {
    "Baseline":               dict(a_s=0.0,    jnei=0.15,     G=G_WP),
    "LSD/DMT (5-HT2A up)":     dict(a_s=GAIN_S, jnei=0.15,     G=G_WP),
    "N2O faithful (disinhib)": dict(a_s=0.0,    jnei=JNEI_n2o, G=G_n2o),
    "COMBO (both)":            dict(a_s=GAIN_S, jnei=JNEI_n2o, G=G_n2o),
}

rows = []
for name, p in conds.items():
    a_e = A_E0 * (1.0 + p["a_s"] * dens)
    J_i = fic_for(a_e, p["jnei"], p["G"])          # rate-match THIS condition to 3 Hz
    sim = build_sim(a_e, p["jnei"], J_i, p["G"])
    S_e = run_sim(sim, 40000.0, discard_ms=15000.0)
    lzc, integ = metrics(S_e)
    rows.append(dict(Condition=name, LZc=round(lzc, 4), Integration=round(integ, 4),
                     rE_Hz=round(s_e_to_rate(S_e.mean()), 2)))
    print("  [%s] rate-matched to %.2f Hz" % (name, s_e_to_rate(S_e.mean())))

df = pd.DataFrame(rows)
df["dLZc_%"] = (100 * (df.LZc / df.LZc[0] - 1)).round(1)
df["dInteg_%"] = (100 * (df.Integration / df.Integration[0] - 1)).round(1)
print(df.to_string(index=False))

combo = df[df.Condition.str.startswith("COMBO")].iloc[0]
n2o = df[df.Condition.str.startswith("N2O")].iloc[0]
lsd = df[df.Condition.str.startswith("LSD")].iloc[0]
print("\n[rate-matched, all ~3 Hz]  LSD dLZc=%s%%  N2O dLZc=%s%%  COMBO dLZc=%s%% dInteg=%s%%"
      % (lsd["dLZc_%"], n2o["dLZc_%"], combo["dLZc_%"], combo["dInteg_%"]))
hit = combo["dLZc_%"] > 0 and combo["dInteg_%"] < 0
print("[hypothesis @ fixed rate] high LZc + collapsed integration: %s"
      % ("SUPPORTED" if hit else "not supported -- earlier 'support' was a rate/hot artifact"))
df.to_csv(r"E:\BiniruProjects\psyche-sim\results_tvb_ratematch.csv", index=False)
print("Saved: results_tvb_ratematch.csv")
