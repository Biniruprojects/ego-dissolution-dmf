"""Download the empirical Beliveau (2017) Cimbi-36 5-HT2A PET map from the netneurolab/
hansen_receptors GitHub repo (the same atlas Deco 2018 used) and sample it at the TVB region
MNI centroids -> real per-region 5-HT2A vector. Saves ht2a_76.npy / ht2a_192.npy (norm 0..1)."""
import numpy as np
import nibabel as nib
import requests

URL = ("https://raw.githubusercontent.com/netneurolab/hansen_receptors/main/"
       "data/PET_nifti_images/5HT2a_cimbi_hc29_beliveau.nii")
path = r"E:\BiniruProjects\psyche-sim\5HT2a_cimbi_beliveau.nii"

r = requests.get(URL, timeout=180, headers={"User-Agent": "Mozilla/5.0"})
print("download status:", r.status_code, " bytes:", len(r.content))
r.raise_for_status()
open(path, "wb").write(r.content)

img = nib.load(path)
data = np.asarray(img.get_fdata(), dtype=float)
data = np.squeeze(data)
inv = np.linalg.inv(img.affine)
print("volume shape:", data.shape, " affine diag:", np.round(np.diag(img.affine), 2),
      " nonzero frac: %.2f" % np.mean(data > 0))

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
    n_nan = int(np.isnan(dens).sum())
    if n_nan:
        dens[np.isnan(dens)] = np.nanmedian(dens)
    dn = (dens - dens.min()) / (dens.max() - dens.min() + 1e-12)
    np.save(r"E:\BiniruProjects\psyche-sim\ht2a_%s.npy" % tag, dn)
    print("\n[%s regions] real 5-HT2A (Cimbi-36) saved (%d centroids needed median fill)." % (tag, n_nan))
    show = ["PFCDL", "PFCM", "PFCORB", "CC", "FEF", "TCPOL", "V1", "V2", "S1", "A1", "M1", "AMYG", "HC"]
    print("   region-class means (normalised 0..1, expect association>>sensory):")
    for s in show:
        idx = [i for i, l in enumerate(labels) if l.endswith(s)]
        if idx:
            print("     %-7s %.2f" % (s, dn[idx].mean()))
print("\nDone.")
