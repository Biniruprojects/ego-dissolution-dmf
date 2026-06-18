"""
Probe the installed TVB API so the driver is written against reality, not guesses.
Prints: version, the ReducedWongWangExcInh model's real parameters, default connectivity,
and whether a feedback-inhibition-control (FIC) helper ships with this version.
"""
import numpy as np

import tvb
print("=== TVB", getattr(tvb, "__version__", "?"), "===")

from tvb.simulator.lab import *  # noqa

# --- default connectivity (does tvb-data ship one?) ---
try:
    conn = connectivity.Connectivity.from_file()
    conn.configure()
    print("[conn] regions=%d  weights=%s  tracts=%s" %
          (conn.number_of_regions, conn.weights.shape, conn.tract_lengths.shape))
    print("[conn] weight range: %.3g .. %.3g" % (conn.weights.min(), conn.weights.max()))
except Exception as e:
    print("[conn] default load FAILED:", repr(e))

# --- the E/I reduced Wong-Wang model ---
try:
    m = models.ReducedWongWangExcInh()
    print("\n[model] ReducedWongWangExcInh")
    pub = [p for p in dir(m) if not p.startswith("_")]
    print("[model] public attrs:", pub)
    for a in ["w_p", "J_N", "J_i", "W_e", "W_i", "I_o", "I_ext", "G",
              "a_e", "b_e", "d_e", "a_i", "b_i", "d_i",
              "gamma_e", "gamma_i", "tau_e", "tau_i", "lamda"]:
        v = getattr(m, a, "N/A")
        print("   %-8s = %s" % (a, getattr(v, "tolist", lambda: v)() if hasattr(v, "tolist") else v))
except Exception as e:
    print("[model] FAILED:", repr(e))

# --- look for a FIC helper anywhere obvious ---
print("\n[FIC] scanning for feedback-inhibition-control helpers...")
hits = []
try:
    import tvb.simulator.models as _mods
    hits += [("models." + n) for n in dir(_mods) if "inhib" in n.lower() or "fic" in n.lower()]
except Exception as e:
    print("   models scan err:", repr(e))
for modname in ["tvb.simulator.integrators", "tvb.analyzers.fmri_balloon", "tvb.contrib"]:
    try:
        mod = __import__(modname, fromlist=["x"])
        hits += [(modname + "." + n) for n in dir(mod) if "inhib" in n.lower() or "fic" in n.lower()]
    except Exception:
        pass
print("   candidates:", hits if hits else "none found by name (will tune FIC manually against TVB model)")

print("\n[ok] probe complete")
