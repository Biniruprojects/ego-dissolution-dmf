"""
Spoor A - v6: add a BOLD monitor (Balloon-Windkessel) on top of the validated DMF+FIC at the
corrected working point. The synaptic-gating FC was ~0.006 (too low to anchor "integration
collapse"); BOLD integrates neural activity over seconds via the haemodynamic response, giving
FC in the realistic ~0.1-0.4 range -- the standard substrate for whole-brain integration claims.

Two monitors run together:
  * TemporalAverage (10 ms)  -> fast S_e, for Lempel-Ziv complexity (entropy axis).
  * Bold (TR 2000 ms)        -> haemodynamic signal, for functional connectivity (integration axis).

Working point: G = 0.06 (just below the stability edge for the empirical connectome, row-sums ~13),
analytic FIC + low-fixed-point init -> baseline clamped at ~3 Hz. Moderate drug doses (no runaway).
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tvb.simulator.lab import *  # noqa

rng = np.random.default_rng(7)

conn = connectivity.Connectivity.from_file()
conn.weights = conn.weights / conn.weights.max()
conn.speed = np.array([3.0])
conn.configure()
N = conn.number_of_regions
print("[conn] empirical, regions=%d, max delay~%.0f ms" % (N, conn.tract_lengths.max() / conn.speed[0]))

dens = np.linspace(0.1, 1.0, N)
rng.shuffle(dens)

A_E0 = 310.0
S_E_TARGET = 0.164
GAIN_S = 0.15
NMDA_BLOCK = 0.3
G_WP = 0.06
NSIG = 1e-4

def s_e_to_rate(s):
    return s / (100.0 * 0.000641 * (1.0 - s))

def build_sim(a_e, J_N, J_i, G=G_WP, dt=1.0, nsig=NSIG, use_bold=False):
    m = models.ReducedWongWangExcInh()
    m.G = np.array([float(G)])
    m.J_N = np.array([float(J_N)])
    m.a_e = np.asarray(a_e, float).reshape(N)
    m.J_i = np.asarray(J_i, float).reshape(N)
    m.variables_of_interest = ['S_e']
    ic = np.zeros((200, 2, N, 1))
    ic[:, 0, :, 0] = 0.164
    ic[:, 1, :, 0] = 0.04
    mons = [monitors.TemporalAverage(period=10.0)]
    if use_bold:
        mons.append(monitors.Bold(period=2000.0))
    sim = simulator.Simulator(
        model=m, connectivity=conn,
        coupling=coupling.Linear(a=np.array([1.0])),
        integrator=integrators.HeunStochastic(dt=dt, noise=noise.Additive(nsig=np.array([nsig]))),
        monitors=mons,
        initial_conditions=ic,
    )
    sim.configure()
    return sim

def run_fast(sim, T, discard_ms=0.0):
    out = sim.run(simulation_length=T)
    t, d = out[0]
    S_e = d[:, 0, :, 0]
    if discard_ms > 0:
        S_e = S_e[int(discard_ms / 10.0):]
    return S_e

def run_both(sim, T, discard_fast=15000.0, discard_bold=20000.0):
    out = sim.run(simulation_length=T)
    S_e = out[0][1][:, 0, :, 0]
    BOLD = out[1][1][:, 0, :, 0]
    S_e = S_e[int(discard_fast / 10.0):]
    BOLD = BOLD[int(discard_bold / 2000.0):]
    return S_e, BOLD

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
    a_e = np.full(N, A_E0)
    it = 0
    for it in range(refine):
        sim = build_sim(a_e, 0.15, J_i, G=G)
        m = run_fast(sim, T, discard_ms=discard).mean(0)
        err = m - S_E_TARGET
        if np.max(np.abs(err)) < 0.012:
            break
        J_i = np.clip(J_i + np.clip(gain * err, -cap, cap), 0.001, None)
    sim = build_sim(a_e, 0.15, J_i, G=G)
    m = run_fast(sim, T, discard_ms=discard).mean(0)
    print("[FIC] S_I*=%.4f, +%d polish: <S_e>=%.3f, r_e~%.2f Hz, maxdev=%.3f"
          % (S_I, it + 1, m.mean(), s_e_to_rate(m.mean()), np.max(np.abs(m - S_E_TARGET))))
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

def lz_metric(S_e):
    Sc = S_e[::5]
    return np.array([lz76(Sc[:, j] > np.median(Sc[:, j])) for j in range(N)])

def fc_metric(BOLD):
    Bz = (BOLD - BOLD.mean(0)) / (BOLD.std(0) + 1e-9)
    FC = np.corrcoef(Bz.T)
    np.fill_diagonal(FC, np.nan)
    nodal = np.nanmean(FC, axis=1)
    iu = np.triu_indices(N, 1)
    return float(np.nanmean(FC[iu])), nodal

print("[setup] FIC at G=%.2f (analytic + polish) ..." % G_WP)
J_i = fic_tune(G=G_WP)

conds = {
    "Baseline":           dict(gain_s=0.0,    block=0.0),
    "LSD/DMT (5-HT2A up)": dict(gain_s=GAIN_S, block=0.0),
    "N2O (NMDA block)":   dict(gain_s=0.0,    block=NMDA_BLOCK),
    "COMBO (both)":       dict(gain_s=GAIN_S, block=NMDA_BLOCK),
}

rows = []
NODE = {}
for name, p in conds.items():
    a_e = A_E0 * (1.0 + p["gain_s"] * dens)
    J_N = 0.15 * (1.0 - p["block"])
    sim = build_sim(a_e, J_N, J_i, G=G_WP, use_bold=True)
    S_e, BOLD = run_both(sim, 150000.0)
    lz_node = lz_metric(S_e)
    integ, nodal = fc_metric(BOLD)
    NODE[name] = (lz_node, nodal)
    rows.append(dict(Condition=name, LZc=round(float(lz_node.mean()), 4),
                     BOLD_FC=round(integ, 4), S_e=round(float(S_e.mean()), 3),
                     rE_Hz=round(s_e_to_rate(S_e.mean()), 2), nBOLD=BOLD.shape[0]))

df = pd.DataFrame(rows)
df["dLZc_%"] = (100 * (df.LZc / df.LZc[0] - 1)).round(1)
df["dFC_%"] = (100 * (df.BOLD_FC / df.BOLD_FC[0] - 1)).round(1)
print(df.to_string(index=False))

lsd = df[df.Condition.str.startswith("LSD")].iloc[0]
base = df.iloc[0]
combo = df[df.Condition.str.startswith("COMBO")].iloc[0]
val_ok = lsd["dLZc_%"] > 0
print("\n[validation] single 5-HT2A-up: dLZc = %s %%  -> %s" % (
    lsd["dLZc_%"], "PASS" if val_ok else "FAIL"))
print("[baseline]   BOLD-FC = %.3f  (realistic resting FC ~0.1-0.4?)" % base.BOLD_FC)
print("[combo]      dLZc = %s %% , dFC = %s %%" % (combo["dLZc_%"], combo["dFC_%"]))
hit = val_ok and base.BOLD_FC > 0.05 and combo["dLZc_%"] > 0 and combo["dFC_%"] < 0
print("[hypothesis] high LZc + collapsed integration: %s" % ("SUPPORTED" if hit else "not (yet) supported"))

deg = conn.weights.sum(axis=1)
bl, bn = NODE["Baseline"]
cl, cn = NODE["COMBO (both)"]
def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))
print("\n[hubs] corr(strength, dLZc)=%+.2f  corr(strength, dFC)=%+.2f" % (corr(deg, cl - bl), corr(deg, cn - bn)))

df.to_csv(r"E:\BiniruProjects\psyche-sim\results_tvb_bold.csv", index=False)
fig, ax = plt.subplots(figsize=(6, 5))
for _, r in df.iterrows():
    ax.scatter(r.BOLD_FC, r.LZc, s=120)
    ax.annotate(r.Condition, (r.BOLD_FC, r.LZc), fontsize=8, xytext=(6, 6), textcoords="offset points")
ax.set_xlabel("BOLD FC integration  (dissociation <-)")
ax.set_ylabel("Lempel-Ziv complexity  (entropy ->)")
ax.set_title("TVB DMF+FIC + BOLD (v6)")
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(r"E:\BiniruProjects\psyche-sim\combo_map_tvb_bold.png", dpi=130)
print("Saved: results_tvb_bold.csv + combo_map_tvb_bold.png")
