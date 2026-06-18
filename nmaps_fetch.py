"""Fetch the empirical Beliveau et al. (2017) 5-HT2A PET density map via neuromaps and sample
it at the TVB region centres (MNI coordinates), producing a real per-region 5-HT2A vector to
replace the synthetic gradient. Saves ht2a_76.npy / ht2a_192.npy (normalised 0..1)."""
import numpy as np
import nibabel as nib
from neuromaps.datasets import fetch_annotation, available_annotations

# discover the Beliveau 5-HT2A annotation
print("available beliveau2017 annotations:")
for a in available_annotations(source="beliveau2017"):
    print("  ", a)

# Beliveau-2017 5-HT2A is the [11C]Cimbi-36 tracer map ('cimbi36'), MNI152 1mm.
ann = None
for attempt in range(6):
    try:
        a = fetch_annotation(source="beliveau2017", desc="cimbi36", space="MNI152", den="1mm")
        if a:
            ann = a
            print("fetched 5-HT2A (Cimbi-36) on attempt %d:" % (attempt + 1), a)
            break
    except Exception as e:
        print("attempt %d failed: %s" % (attempt + 1, type(e).__name__))
if not ann:
    raise SystemExit("OSF still unreachable after retries; the Cimbi-36 map could not be downloaded")

path = ann if isinstance(ann, str) else (list(ann.values())[0] if isinstance(ann, dict) else ann[0])
img = nib.load(str(path))
data = np.asarray(img.get_fdata(), dtype=float)
inv = np.linalg.inv(img.affine)
print("map shape", data.shape, " nonzero frac %.2f" % (np.mean(data > 0)))

def sample(xyz):
    v = inv @ np.array([xyz[0], xyz[1], xyz[2], 1.0])
    i, j, k = [int(round(t)) for t in v[:3]]
    if not (0 <= i < data.shape[0] and 0 <= j < data.shape[1] and 0 <= k < data.shape[2]):
        return np.nan
    sub = data[max(i-2, 0):i+3, max(j-2, 0):j+3, max(k-2, 0):k+3]
    sub = sub[np.isfinite(sub) & (sub > 0)]
    return sub.mean() if sub.size else np.nan

for tag in ["76", "192"]:
    centres = np.load(r"E:\BiniruProjects\psyche-sim\centres_%s.npy" % tag)
    labels = np.load(r"E:\BiniruProjects\psyche-sim\labels_%s.npy" % tag)
    dens = np.array([sample(c) for c in centres])
    if np.isnan(dens).any():
        dens[np.isnan(dens)] = np.nanmedian(dens)
    dn = (dens - dens.min()) / (dens.max() - dens.min() + 1e-12)   # normalise 0..1
    np.save(r"E:\BiniruProjects\psyche-sim\ht2a_%s.npy" % tag, dn)
    print("\n[%s regions] real 5-HT2A density (normalised) saved." % tag)
    # sanity: association cortex should be high, primary sensory low
    show = ["PFCDL", "PFCM", "PFCORB", "CC", "TCPOL", "V1", "V2", "S1", "A1", "M1"]
    for s in show:
        idx = [i for i, l in enumerate(labels) if l.endswith(s)]
        if idx:
            print("   %-7s mean=%.2f" % (s, dn[idx].mean()))
print("\nDone.")
