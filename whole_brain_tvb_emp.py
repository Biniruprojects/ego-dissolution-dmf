"""
Spoor A - v4b: validated DMF+FIC via TVB, now on the EMPIRICAL connectome with real
conduction delays (first hardening step from whole_brain_tvb.py's synthetic POC).

Changes vs whole_brain_tvb.py:
  * connectome = TVB default empirical (Hagmann 76-region), weights normalised to max=1.
  * real tract lengths + finite conduction speed (3 mm/ms) -> distance-dependent delays
    (up to ~50 ms). Delays + real topology are the standard ingredients for realistic
    resting-state FC, which the synthetic POC lacked (baseline FC was ~0.017).
Everything else identical: 5-HT2A = a_e gain up (density-weighted), N2O = J_N down,
FIC tunes J_i to <S_e>~0.164 (r_e~3 Hz), held fixed under acute drug.

5-HT2A density map is still SYNTHETIC (real PET atlas = next hardening step).
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

# --- empirical connectome + real delays ---
conn = connectivity.Connectivity.from_file()
conn.weights = conn.weights / conn.weights.max()     # normalise (G absorbs scale)
conn.speed = np.array([3.0])                          # mm/ms -> delays from tract_lengths
conn.configure()
N = conn.number_of_regions
print("[conn] empirical, regions=%d, max delay~%.0f ms" %
      (N, conn.tract_lengths.max() / conn.speed[0]))

# --- synthetic 5-HT2A density map (real PET atlas = later hardening) ---
dens = np.linspace(0.1, 1.0, N)
rng.shuffle(dens)

A_E0 = 310.0
S_E_TARGET = 0.164
GAIN_S = 0.15                             # 5-HT2A: gentle +15% gain (stay near-critical, no runaway)
NMDA_BLOCK = 0.3                          # N2O: moderate -30% NMDA current
G_WP = 0.06                               # just below the stability edge for THIS connectome
                                          # (empirical row-sums ~13, so the working G is ~0.06,
                                          #  not 2.0; max FC sits at the edge of criticality)
NSIG = 1e-4

def s_e_to_rate(s):
    return s / (100.0 * 0.000641 * (1.0 - s))

def build_sim(a_e, J_N, J_i, G=G_WP, dt=1.0, nsig=NSIG, period=10.0):
    m = models.ReducedWongWangExcInh()
    m.G = np.array([float(G)])
    m.J_N = np.array([float(J_N)])
    m.a_e = np.asarray(a_e, float).reshape(N)
    m.J_i = np.asarray(J_i, float).reshape(N)
    # initialise at the LOW (3 Hz) fixed point so the bistable node stays in that basin
    # instead of sliding into the hot attractor (which the analytic J_i does not hold).
    ic = np.zeros((200, 2, N, 1))
    ic[:, 0, :, 0] = 0.164
    ic[:, 1, :, 0] = 0.04
    sim = simulator.Simulator(
        model=m, connectivity=conn,
        coupling=coupling.Linear(a=np.array([1.0])),
        integrator=integrators.HeunStochastic(dt=dt, noise=noise.Additive(nsig=np.array([nsig]))),
        monitors=[monitors.TemporalAverage(period=period)],
        initial_conditions=ic,
    )
    sim.configure()
    return sim

def run_sim(sim, T, discard_ms=0.0):
    (t, d), = sim.run(simulation_length=T)
    S_e = d[:, 0, :, 0]
    if discard_ms > 0:
        S_e = S_e[int(discard_ms / 10.0):]
    return S_e

def H_I_scalar(x):
    # inhibitory f-I curve with exp-overflow guard (stiff below threshold)
    y = 615.0 * x - 177.0
    ey = np.exp(min(-0.087 * y, 50.0))
    return y / (1.0 - ey)

def analytic_fic(G):
    # Deco 2014 ANALYTIC FIC: at the set-point every node sits at S_e = 0.164 (r_e ~ 3.06 Hz),
    # so the operating point is fully known and J_i solves in CLOSED FORM per node -- no
    # stochastic iteration needs to converge. I_E_target = (b_e - 8)/a_e = 0.377 nA, the
    # famous -0.026 nA set-point. Hubs (high strength k_i) get their high J_i assigned directly.
    aI, bI, dI = 615.0, 177.0, 0.087  # noqa (documented; used inside H_I_scalar)
    gI, tI = 0.001, 10.0
    wp, JN, We, Wi, Io = 1.4, 0.15, 1.0, 0.7, 0.382
    S_E = 0.164
    I_E_target = (125.0 - 8.0) / 310.0            # 0.3774 nA -> r_e ~ 3.06 Hz
    # inhibitory fixed point S_I*: root of  tI*gI*H_I(Wi*Io + JN*S_E - S_I) - S_I  (bisection)
    def g(S_I):
        return tI * gI * H_I_scalar(Wi * Io + JN * S_E - S_I) - S_I
    lo, hi = 1e-4, 0.20
    glo = g(lo)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if glo * g(mid) <= 0:
            hi = mid
        else:
            lo = mid
            glo = g(lo)
    S_I = 0.5 * (lo + hi)
    k = conn.weights.sum(axis=1)                  # node strength (normalised weights)
    J_i = (We * Io + wp * JN * S_E + G * JN * S_E * k - I_E_target) / S_I
    return np.clip(J_i, 0.001, None), S_I

def fic_tune(G=G_WP, refine=25, gain=3.0, cap=0.5, T=2000.0, discard=1000.0):
    # analytic seed (near-exact) + light stochastic polish for noise/delay/finite-size
    J_i, S_I = analytic_fic(G)
    a_e = np.full(N, A_E0)
    it = 0
    for it in range(refine):
        sim = build_sim(a_e, 0.15, J_i, G=G)
        m = run_sim(sim, T, discard_ms=discard).mean(0)
        err = m - S_E_TARGET
        if np.max(np.abs(err)) < 0.012:
            break
        J_i = np.clip(J_i + np.clip(gain * err, -cap, cap), 0.001, None)
    sim = build_sim(a_e, 0.15, J_i, G=G)
    m = run_sim(sim, T, discard_ms=discard).mean(0)
    print("[FIC-analytic] S_I*=%.4f, +%d polish iters: <S_e>=%.3f (target %.3f), r_e~%.2f Hz, maxdev=%.3f"
          % (S_I, it + 1, m.mean(), S_E_TARGET, s_e_to_rate(m.mean()), np.max(np.abs(m - S_E_TARGET))))
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
    return lzc, FC[iu].mean(), FC

def node_metrics(S_e):
    # per-node LZc and per-node nodal integration (mean FC to the rest of the brain)
    Sc = S_e[::5]
    lz = np.array([lz76(Sc[:, j] > np.median(Sc[:, j])) for j in range(N)])
    Sz = (S_e - S_e.mean(0)) / (S_e.std(0) + 1e-9)
    FC = np.corrcoef(Sz.T)
    np.fill_diagonal(FC, np.nan)
    ni = np.nanmean(FC, axis=1)
    return lz, ni

print("[setup] tuning FIC on empirical connectome, G=%.1f ..." % G_WP)
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
    sim = build_sim(a_e, J_N, J_i, G=G_WP)
    S_e = run_sim(sim, 40000.0, discard_ms=15000.0)
    lzc, integ, FC = metrics(S_e)
    NODE[name] = node_metrics(S_e)
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
print("[baseline]   integration = %.3f" % base.Integration)
print("[combo]      dLZc = %s %% , dInteg = %s %%" % (combo["dLZc_%"], combo["dInteg_%"]))
hit = val_ok and base.Integration > 0.05 and combo["dLZc_%"] > 0 and combo["dInteg_%"] < 0
print("[hypothesis] high LZc + collapsed integration: %s" % ("SUPPORTED" if hit else "not (yet) supported"))

# ---------- hub analysis: do high-strength nodes AMPLIFY or locally BREAK the cancellation? ----------
deg = conn.weights.sum(axis=1)                 # weighted node strength (hubness)
base_lz, base_ni = NODE["Baseline"]
combo_lz, combo_ni = NODE["COMBO (both)"]
dlz = combo_lz - base_lz                        # per-node LZc change in the combo
dni = combo_ni - base_ni                        # per-node nodal-integration change in the combo
def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))
hi = deg >= np.median(deg)
print("\n[hubs] does real human topology amplify or break the mutual cancellation?")
print("   corr(strength, dLZc_node)      = %+.2f" % corr(deg, dlz))
print("   corr(strength, d_nodal_integ)  = %+.2f" % corr(deg, dni))
print("   hubs  (top 50%% strength): mean dLZc=%+.4f  mean dInteg=%+.4f" % (dlz[hi].mean(), dni[hi].mean()))
print("   non-hubs                : mean dLZc=%+.4f  mean dInteg=%+.4f" % (dlz[~hi].mean(), dni[~hi].mean()))
verdict = "BREAK locally at hubs" if abs(corr(deg, dlz)) > 0.3 or np.sign(dlz[hi].mean()) != np.sign(dlz[~hi].mean()) \
    else "uniform (hubs do NOT break it)"
print("   -> %s" % verdict)

df.to_csv(r"E:\BiniruProjects\psyche-sim\results_tvb_emp.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 5))
for _, r in df.iterrows():
    ax.scatter(r.Integration, r.LZc, s=120)
    ax.annotate(r.Condition, (r.Integration, r.LZc), fontsize=8,
                xytext=(6, 6), textcoords="offset points")
ax.set_xlabel("FC integration  (dissociation <-)")
ax.set_ylabel("Lempel-Ziv complexity  (entropy ->)")
ax.set_title("TVB DMF+FIC, empirical connectome (v4b)")
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig(r"E:\BiniruProjects\psyche-sim\combo_map_tvb_emp.png", dpi=130)
print("Saved: results_tvb_emp.csv + combo_map_tvb_emp.png")
