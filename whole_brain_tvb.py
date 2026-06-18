"""
Spoor A - v4: VALIDATED Dynamic Mean Field (reduced Wong-Wang, Deco 2014) + Feedback
Inhibition Control, via The Virtual Brain (tvb-library 2.10). TVB's tested numba dfun +
HeunStochastic integrator replace the hand-rolled Euler that kept collapsing/exploding in
v3 -- the model equations are bistable, so a validated integrator + the correct FIC target
are what land the realistic working point.

Knobs (read straight from tvb/simulator/models/wong_wang_exc_inh.py):
  * 5-HT2A agonism (LSD/DMT) -> a_e (excitatory input GAIN) up, per-region, weighted by the
       5-HT2A density map.  (Deco et al. 2018 serotonin-map method; a_e is the gain slope.)
  * NMDA block (N2O)         -> J_N (NMDA current) down. J_N enters local recurrence
       (w_p*J_N*S_e), long-range coupling (G*J_N*c) AND E->I drive -> faithful NMDA
       antagonism.
  * FIC                      -> J_i (local inhibition) per region, tuned so <S_e> ~ 0.164,
       i.e. excitatory rate ~ 3 Hz (the Deco set-point). Tuned at baseline, held fixed under
       the acute drug perturbations.

Coupling wiring: the model itself computes G*J_N*c_0 (source line 246), so we use
coupling.Linear(a=1.0) and set G on the model -- no double scaling.

Validation gate: single 5-HT2A-up must raise LZc (Schartner 2017) AND baseline integration
must be substantially > 0, before the combo quadrant counts.

Connectome + 5-HT2A map still SYNTHETIC (POC). Hardening = empirical connectome (tvb-data /
HCP) + real 5-HT2A PET atlas (Beliveau/Hansen).
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tvb.simulator.lab import *  # noqa

rng = np.random.default_rng(7)
N = 90

# --- synthetic small-world connectome (symmetric, no self-loops, scaled to [0,1]) ---
Gsw = nx.watts_strogatz_graph(N, k=8, p=0.3, seed=7)
C = nx.to_numpy_array(Gsw)
C *= rng.uniform(0.5, 1.5, C.shape)
C = (C + C.T) / 2
np.fill_diagonal(C, 0.0)
C /= C.max()

# --- synthetic 5-HT2A density map (assoc-cortex-like gradient, 0..1) ---
dens = np.linspace(0.1, 1.0, N)
rng.shuffle(dens)

conn = connectivity.Connectivity(
    weights=C,
    tract_lengths=np.zeros((N, N)),
    region_labels=np.array(["R%02d" % i for i in range(N)]),
    centres=np.zeros((N, 3)),
)
conn.speed = np.array([np.inf])          # negligible delays for this POC
conn.configure()

A_E0 = 310.0                              # baseline excitatory gain
S_E_TARGET = 0.164                        # <S_e> at r_e ~ 3 Hz (FIC set-point)
GAIN_S = 0.4                              # 5-HT2A: a_e up to +40% where density=1
NMDA_BLOCK = 0.5                          # N2O: J_N halved
G_WP = 2.0                                # global coupling working point
NSIG = 1e-4                               # noise amplitude

def s_e_to_rate(s):
    # invert the S_e steady-state relation to report an approximate firing rate (Hz)
    return s / (100.0 * 0.000641 * (1.0 - s))

def build_sim(a_e, J_N, J_i, G=G_WP, dt=1.0, nsig=NSIG, period=10.0):
    m = models.ReducedWongWangExcInh()
    m.G = np.array([float(G)])
    m.J_N = np.array([float(J_N)])
    m.a_e = np.asarray(a_e, float).reshape(N)     # per-region 5-HT2A gain
    m.J_i = np.asarray(J_i, float).reshape(N)     # per-region FIC inhibition
    sim = simulator.Simulator(
        model=m, connectivity=conn,
        coupling=coupling.Linear(a=np.array([1.0])),
        integrator=integrators.HeunStochastic(dt=dt, noise=noise.Additive(nsig=np.array([nsig]))),
        monitors=[monitors.TemporalAverage(period=period)],
    )
    sim.configure()
    return sim

def run_sim(sim, T, discard_ms=0.0):
    (t, d), = sim.run(simulation_length=T)
    S_e = d[:, 0, :, 0]
    if discard_ms > 0:
        S_e = S_e[int(discard_ms / 10.0):]
    return S_e

# ---------- FIC: tune J_i so <S_e> ~ target (warm-continued chunks) ----------
def fic_tune(G=G_WP, iters=120, gain=4.0, cap=0.8, T=2500.0, discard=1200.0):
    # Rebuild-per-iter: each step measures the CLEAN steady-state <S_e> for the current J_i
    # (fresh init + transient discard, no warm-state lag), then a capped proportional update
    # nudges J_i toward the set-point. This converges where the warm-continued version lagged.
    a_e = np.full(N, A_E0)
    J_i = np.ones(N)
    it = 0
    for it in range(iters):
        sim = build_sim(a_e, 0.15, J_i, G=G)
        m = run_sim(sim, T, discard_ms=discard).mean(0)
        err = m - S_E_TARGET
        if np.max(np.abs(err)) < 0.015:
            break
        J_i = np.clip(J_i + np.clip(gain * err, -cap, cap), 0.001, None)
    sim = build_sim(a_e, 0.15, J_i, G=G)
    m = run_sim(sim, T, discard_ms=discard).mean(0)
    print("[FIC] stopped after %d iters: <S_e>=%.3f (target %.3f), r_e~%.2f Hz, maxdev=%.3f"
          % (it + 1, m.mean(), S_E_TARGET, s_e_to_rate(m.mean()), np.max(np.abs(m - S_E_TARGET))))
    return J_i

# ---------- metrics ----------
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
    return lzc, FC[iu].mean(), FC

# ---------- tune FIC at baseline, then run the 4 conditions with J_i fixed ----------
print("[setup] tuning Feedback Inhibition Control (TVB DMF) at baseline, G=%.1f ..." % G_WP)
J_i = fic_tune(G=G_WP)

conds = {
    "Baseline":           dict(gain_s=0.0,     block=0.0),
    "LSD/DMT (5-HT2A up)": dict(gain_s=GAIN_S,  block=0.0),
    "N2O (NMDA block)":   dict(gain_s=0.0,     block=NMDA_BLOCK),
    "COMBO (both)":       dict(gain_s=GAIN_S,  block=NMDA_BLOCK),
}

rows = []
FCs = {}
for name, p in conds.items():
    a_e = A_E0 * (1.0 + p["gain_s"] * dens)
    J_N = 0.15 * (1.0 - p["block"])
    sim = build_sim(a_e, J_N, J_i, G=G_WP)
    S_e = run_sim(sim, 40000.0, discard_ms=15000.0)
    lzc, integ, FC = metrics(S_e)
    FCs[name] = FC
    rows.append(dict(Condition=name, LZc=round(lzc, 4), Integration=round(integ, 4),
                     S_e=round(float(S_e.mean()), 3), rE_Hz=round(s_e_to_rate(S_e.mean()), 2)))

df = pd.DataFrame(rows)
df["dLZc_%"] = (100 * (df.LZc / df.LZc[0] - 1)).round(1)
df["dInteg_%"] = (100 * (df.Integration / df.Integration[0] - 1)).round(1)
print(df.to_string(index=False))

lsd = df[df.Condition.str.startswith("LSD")].iloc[0]
base = df.iloc[0]
combo = df[df.Condition.str.startswith("COMBO")].iloc[0]
val_ok = lsd["dLZc_%"] > 0
print("\n[validation] single 5-HT2A-up: dLZc = %s %%  -> %s" % (
    lsd["dLZc_%"], "PASS (entropy up, Schartner 2017)" if val_ok else "FAIL"))
print("[baseline]   integration = %.3f  (needs to be clearly > 0 for 'collapse' to be real)" % base.Integration)
print("[combo]      dLZc = %s %% , dInteg = %s %%" % (combo["dLZc_%"], combo["dInteg_%"]))
hit = val_ok and base.Integration > 0.05 and combo["dLZc_%"] > 0 and combo["dInteg_%"] < 0
print("[hypothesis] high LZc + collapsed integration: %s" % ("SUPPORTED" if hit else "not (yet) supported"))

df.to_csv(r"E:\BiniruProjects\psyche-sim\results_tvb.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 5))
for _, r in df.iterrows():
    ax.scatter(r.Integration, r.LZc, s=120)
    ax.annotate(r.Condition, (r.Integration, r.LZc), fontsize=8,
                xytext=(6, 6), textcoords="offset points")
ax.set_xlabel("FC integration  (dissociation <-)")
ax.set_ylabel("Lempel-Ziv complexity  (entropy ->)")
ax.set_title("TVB DMF+FIC (v4) - where the COMBO lands")
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(r"E:\BiniruProjects\psyche-sim\combo_map_tvb.png", dpi=130)
print("Saved: results_tvb.csv + combo_map_tvb.png")
