"""
Spoor A - v7: FAITHFUL N2O mechanism. The decisive test of the combo hypothesis.

Until now N2O was modelled crudely as a net J_N reduction -> less excitation -> LOWER complexity,
which opposed the 5-HT2A entropy boost so the combo cancelled. But sub-anaesthetic NMDA antagonism
(N2O, ketamine) is INTERNEURON-PREFERENTIAL: it blocks NMDA on inhibitory interneurons first ->
DISINHIBITION -> cortical excitation and gamma/complexity UP. Plus it weakens long-range NMDA
coupling -> dissociation / integration DOWN.

We model this by subclassing the validated reduced Wong-Wang DMF with a SEPARATE excitatory->
inhibitory conductance J_N_EI (the interneuron-NMDA target). Faithful N2O then does two things:
  * J_N_EI down   -> less drive to inhibition -> disinhibition -> E activity / complexity UP
  * G down        -> weaker long-range coupling -> integration DOWN (dissociation)

Hypothesis test: does a faithful (disinhibitory) N2O, combined with 5-HT2A gain, finally produce
HIGH complexity + COLLAPSED integration -- where the crude N2O could not?
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

dens = np.linspace(0.1, 1.0, N)
rng.shuffle(dens)

A_E0 = 310.0
S_E_TARGET = 0.164
GAIN_S = 0.15        # 5-HT2A gain
BLOCK_EI = 0.4       # N2O interneuron NMDA block -> J_N_EI *= (1-BLOCK_EI)  (disinhibition)
BLOCK_G = 0.3        # N2O long-range NMDA reduction -> G *= (1-BLOCK_G)     (dissociation)
G_WP = 0.06
NSIG = 1e-4

class N2OWongWang(models.ReducedWongWangExcInh):
    """reduced Wong-Wang with a separate excitatory->inhibitory NMDA conductance J_N_EI,
    so interneuron-preferential NMDA block (disinhibition) can be modelled apart from the
    excitatory recurrence / long-range coupling."""
    J_N_EI = NArray(default=np.array([0.15]), doc="E->I NMDA conductance (interneuron block target)")

    def dfun(self, x, c, local_coupling=0.0, **kwargs):
        S = x
        ae = np.reshape(self.a_e, (-1, 1))
        ji = np.reshape(self.J_i, (-1, 1))
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

def build_sim(a_e, J_N_EI, J_i, G=G_WP, dt=1.0, nsig=NSIG):
    m = N2OWongWang()
    m.G = np.array([float(G)])
    m.J_N = np.array([0.15])
    m.J_N_EI = np.array([float(J_N_EI)])
    m.a_e = np.asarray(a_e, float).reshape(N)
    m.J_i = np.asarray(J_i, float).reshape(N)
    m.variables_of_interest = ['S_e']
    ic = np.zeros((200, 2, N, 1))
    ic[:, 0, :, 0] = 0.164
    ic[:, 1, :, 0] = 0.04
    sim = simulator.Simulator(
        model=m, connectivity=conn,
        coupling=coupling.Linear(a=np.array([1.0])),
        integrator=integrators.HeunStochastic(dt=dt, noise=noise.Additive(nsig=np.array([nsig]))),
        monitors=[monitors.TemporalAverage(period=10.0)],
        initial_conditions=ic)
    sim.configure()
    return sim

def run_sim(sim, T, discard_ms=0.0):
    t, d = sim.run(simulation_length=T)[0]
    S_e = d[:, 0, :, 0]
    if discard_ms > 0:
        S_e = S_e[int(discard_ms / 10.0):]
    return S_e

def H_I_scalar(x):
    y = 615.0 * x - 177.0
    ey = np.exp(min(-0.087 * y, 50.0))
    return y / (1.0 - ey)

def analytic_fic(G):
    gI, tI = 0.001, 10.0
    wp, JN, We, Wi, Io = 1.4, 0.15, 1.0, 0.7, 0.382
    S_E = 0.164
    I_E_target = (125.0 - 8.0) / 310.0
    def g(S_I):
        return tI * gI * H_I_scalar(Wi * Io + JN * S_E - S_I) - S_I
    lo, hi = 1e-4, 0.20
    glo = g(lo)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if glo * g(mid) <= 0:
            hi = mid
        else:
            lo = mid; glo = g(lo)
    S_I = 0.5 * (lo + hi)
    k = conn.weights.sum(axis=1)
    J_i = (We * Io + wp * JN * S_E + G * JN * S_E * k - I_E_target) / S_I
    return np.clip(J_i, 0.001, None), S_I

def fic_tune(G=G_WP, refine=25, gain=3.0, cap=0.5, T=2000.0, discard=1000.0):
    J_i, S_I = analytic_fic(G)
    it = 0
    for it in range(refine):
        sim = build_sim(np.full(N, A_E0), 0.15, J_i, G=G)
        m = run_sim(sim, T, discard_ms=discard).mean(0)
        err = m - S_E_TARGET
        if np.max(np.abs(err)) < 0.012:
            break
        J_i = np.clip(J_i + np.clip(gain * err, -cap, cap), 0.001, None)
    sim = build_sim(np.full(N, A_E0), 0.15, J_i, G=G)
    m = run_sim(sim, T, discard_ms=discard).mean(0)
    print("[FIC] +%d polish: <S_e>=%.3f, r_e~%.2f Hz, maxdev=%.3f"
          % (it + 1, m.mean(), s_e_to_rate(m.mean()), np.max(np.abs(m - S_E_TARGET))))
    return J_i

def lz76(b):
    s = ''.join('1' if v else '0' for v in b)
    n = len(s)
    if n < 2:
        return 0.0
    i, c, l = 0, 1, 1
    k, kmax = 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            kmax = max(kmax, k)
            i += 1
            if i == l:
                c += 1
                l += kmax
                if l + 1 > n:
                    break
                i = 0
                k = 1
                kmax = 1
            else:
                k = 1
    return c / (n / np.log2(n))

def metrics(S_e):
    Sc = S_e[::5]
    lzc = np.mean([lz76(Sc[:, j] > np.median(Sc[:, j])) for j in range(N)])
    Sz = (S_e - S_e.mean(0)) / (S_e.std(0) + 1e-9)
    FC = np.corrcoef(Sz.T)
    iu = np.triu_indices(N, 1)
    return lzc, FC[iu].mean()

print("[setup] FIC at G=%.2f ..." % G_WP)
J_i = fic_tune(G=G_WP)

# faithful N2O = disinhibition (J_N_EI down) + dissociation (G down)
JNEI_n2o = 0.15 * (1.0 - BLOCK_EI)
G_n2o = G_WP * (1.0 - BLOCK_G)
conds = {
    "Baseline":                 dict(a_s=0.0,    jnei=0.15,      G=G_WP),
    "LSD/DMT (5-HT2A up)":       dict(a_s=GAIN_S, jnei=0.15,      G=G_WP),
    "N2O faithful (disinhib)":   dict(a_s=0.0,    jnei=JNEI_n2o,  G=G_n2o),
    "COMBO (both)":              dict(a_s=GAIN_S, jnei=JNEI_n2o,  G=G_n2o),
}

rows = []
for name, p in conds.items():
    a_e = A_E0 * (1.0 + p["a_s"] * dens)
    sim = build_sim(a_e, p["jnei"], J_i, G=p["G"])
    S_e = run_sim(sim, 40000.0, discard_ms=15000.0)
    lzc, integ = metrics(S_e)
    rows.append(dict(Condition=name, LZc=round(lzc, 4), Integration=round(integ, 4),
                     S_e=round(float(S_e.mean()), 3), rE_Hz=round(s_e_to_rate(S_e.mean()), 2)))

df = pd.DataFrame(rows)
df["dLZc_%"] = (100 * (df.LZc / df.LZc[0] - 1)).round(1)
df["dInteg_%"] = (100 * (df.Integration / df.Integration[0] - 1)).round(1)
print(df.to_string(index=False))

lsd = df[df.Condition.str.startswith("LSD")].iloc[0]
n2o = df[df.Condition.str.startswith("N2O")].iloc[0]
combo = df[df.Condition.str.startswith("COMBO")].iloc[0]
print("\n[faithful N2O alone] dLZc = %s %% (crude N2O gave NEGATIVE -> does disinhibition flip it UP?)"
      % n2o["dLZc_%"])
print("[validation] 5-HT2A-up dLZc = %s %%  -> %s" % (lsd["dLZc_%"], "PASS" if lsd["dLZc_%"] > 0 else "FAIL"))
print("[combo]      dLZc = %s %% , dInteg = %s %%" % (combo["dLZc_%"], combo["dInteg_%"]))
hit = combo["dLZc_%"] > 0 and combo["dInteg_%"] < 0
print("[hypothesis] high LZc + collapsed integration: %s" % ("SUPPORTED" if hit else "not supported"))
df.to_csv(r"E:\BiniruProjects\psyche-sim\results_tvb_n2o.csv", index=False)
print("Saved: results_tvb_n2o.csv")
