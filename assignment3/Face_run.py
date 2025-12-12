import os
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import ps_utils

# -------- Configuration --------
script_dir = os.path.dirname(os.path.abspath(__file__))
MATFILE = os.path.join(script_dir, "face.mat")             
RANSAC_THRESHOLD = 10.0         
RANSAC_MAX_ITERS = 2000
RANSAC_MAX_DATA_TRIES = 200
SMOOTH_ITERS = 200              
EPS_N3 = 1e-6
FIGS_DIR = "figs_face"
USE_FALLBACK_PINV = True        
VERBOSE = True
# --------------------------------

os.makedirs(FIGS_DIR, exist_ok=True)

def imsave(img, fname, cmap='gray'):
    plt.figure(figsize=(6,5)); plt.imshow(img, cmap=cmap); plt.axis('off')
    plt.tight_layout(); plt.savefig(fname, dpi=200, bbox_inches='tight'); plt.close()

def show_depth_views_multi(Z, mask, outname, views=None, figsize=(18,6)):
    if views is None:
        views = [(30,45),(60,-60),(20,120)]
    Zplot = Z.copy().astype(float)
    Zplot[mask==0] = np.nan
    mZ, nZ = Zplot.shape
    X, Y = np.meshgrid(np.arange(nZ), np.arange(mZ))
    fig = plt.figure(figsize=figsize)
    for i,(elev,azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, len(views), i, projection='3d')
        if np.all(~np.isfinite(Zplot)):
            ax.text(0.5,0.5,0.5,'No valid depth', transform=ax.transAxes, ha='center')
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_title(f"view {i}")
            continue
        ax.plot_surface(X, Y, Zplot, rstride=1, cstride=1, linewidth=0, antialiased=True, cmap='viridis', edgecolor='none')
        ax.view_init(elev=elev, azim=azim)
        ax.set_xticks([]); ax.set_yticks([])
        try:
            zmin = np.nanmin(Zplot); zmax = np.nanmax(Zplot)
            if np.isfinite(zmin) and np.isfinite(zmax):
                ax.set_zlim(zmin, zmax)
        except Exception:
            pass
    plt.tight_layout(); plt.savefig(outname, dpi=200, bbox_inches='tight'); plt.close(fig)

# ---------- Load data ----------
if not os.path.exists(MATFILE):
    raise FileNotFoundError(f"{MATFILE} not found in cwd {os.getcwd()}")

I, mask, S = ps_utils.read_data_file(MATFILE)
m, n, nb = I.shape
print("Image shape:", m, n, "#images:", nb)
print("S shape:", S.shape)

# Build J (nb x nz)
mask_idx = np.where(mask > 0)
nz = mask_idx[0].size
print("nz:", nz)
J = np.zeros((nb, nz), dtype=float)
for k in range(nb):
    J[k,:] = I[:,:,k][mask_idx]

# Compute pinv baseline (useful fallback / comparison)
Spinv = np.linalg.pinv(S)
M_pinv = Spinv @ J         # 3 x nz
albedo_pinv = np.linalg.norm(M_pinv, axis=0)
norms = albedo_pinv.copy(); norms[norms < 1e-8] = 1.0
N_pinv = M_pinv / norms

# reconstruct images for pinv
albedo_img_pinv = np.zeros((m,n)); n1_pinv = np.zeros((m,n)); n2_pinv = np.zeros((m,n)); n3_pinv = np.zeros((m,n))
albedo_img_pinv[mask_idx] = albedo_pinv
n1_pinv[mask_idx] = N_pinv[0,:]; n2_pinv[mask_idx] = N_pinv[1,:]; n3_pinv[mask_idx] = N_pinv[2,:]

# Save baseline albedo
imsave(albedo_img_pinv, os.path.join(FIGS_DIR, "face_pinv_albedo.png"))

# ---------- RANSAC per-pixel ----------
M_ransac = np.zeros_like(M_pinv)
inliers_count = np.zeros(nz, dtype=int)
print("Running RANSAC per-pixel with threshold", RANSAC_THRESHOLD)
t0 = time.time()
for i in range(nz):
    Ivec = J[:,i]
    try:
        res = ps_utils.ransac_3dvector((Ivec, S), threshold=RANSAC_THRESHOLD,
                                       max_data_tries=RANSAC_MAX_DATA_TRIES,
                                       max_iters=RANSAC_MAX_ITERS, p=0.99, verbose=0)
    except Exception:
        res = None
    if res is None:
        if USE_FALLBACK_PINV:
            m_est = M_pinv[:, i]
            inliers = None
        else:
            m_est = np.zeros(3)
            inliers = None
    else:
        m_est, inliers, fit = res
        if inliers is not None:
            inliers_count[i] = len(inliers)
    M_ransac[:, i] = m_est
t1 = time.time()
print("RANSAC done in {:.1f}s".format(t1-t0))
print("Inliers stats (min, median, mean, max):", np.min(inliers_count), np.median(inliers_count), np.mean(inliers_count), np.max(inliers_count))
print("Fraction pixels with >0 inliers:", np.mean(inliers_count>0))

# Build albedo & normals from RANSAC
albedo_ransac = np.linalg.norm(M_ransac, axis=0)
norms = albedo_ransac.copy(); norms[norms < 1e-8] = 1.0
N_ransac = M_ransac / norms

albedo_img_ransac = np.zeros((m,n)); n1_ransac = np.zeros((m,n)); n2_ransac = np.zeros((m,n)); n3_ransac = np.zeros((m,n))
albedo_img_ransac[mask_idx] = albedo_ransac
n1_ransac[mask_idx] = N_ransac[0,:]; n2_ransac[mask_idx] = N_ransac[1,:]; n3_ransac[mask_idx] = N_ransac[2,:]

imsave(albedo_img_ransac, os.path.join(FIGS_DIR, "face_ransac_albedo.png"))

# ---------- Smooth normals ----------
# Choose integration mask where n3 not too small
mask_integrate = (mask>0) & (np.abs(n3_ransac) > EPS_N3)
print("Valid integrate pixels:", np.sum(mask_integrate), "/", nz)

# Prepare arrays for smoothing (set outside region safe defaults)
n1_tmp = n1_ransac.copy(); n2_tmp = n2_ransac.copy(); n3_tmp = n3_ransac.copy()
n1_tmp[~mask_integrate] = 0.0; n2_tmp[~mask_integrate] = 0.0; n3_tmp[~mask_integrate] = 1.0

# Smooth
print("Smoothing normals (iters={})...".format(SMOOTH_ITERS))
n1_sm, n2_sm, n3_sm = ps_utils.smooth_normal_field(n1_tmp, n2_tmp, n3_tmp, mask_integrate.astype(np.uint8), iters=SMOOTH_ITERS, tau=0.05, verbose=False)

# For visualization set outside mask to NaN
n1_sm_vis = n1_sm.copy(); n2_sm_vis = n2_sm.copy(); n3_sm_vis = n3_sm.copy()
n1_sm_vis[~mask_integrate] = np.nan; n2_sm_vis[~mask_integrate] = np.nan; n3_sm_vis[~mask_integrate] = np.nan

# Save normal component visuals (map -1..1 -> 0..1)
imsave((n1_sm_vis+1)/2, os.path.join(FIGS_DIR, "face_ransac_sm_n1.png"))
imsave((n2_sm_vis+1)/2, os.path.join(FIGS_DIR, "face_ransac_sm_n2.png"))
imsave((n3_sm_vis+1)/2, os.path.join(FIGS_DIR, "face_ransac_sm_n3.png"))

# Prepare safe versions for integration (replace outside with safe values)
n1_for_int = n1_sm.copy(); n2_for_int = n2_sm.copy(); n3_for_int = n3_sm.copy()
n1_for_int[~mask_integrate] = 0.0; n2_for_int[~mask_integrate] = 0.0; n3_for_int[~mask_integrate] = 1.0

try:
    z_ransac_sm = ps_utils.simchony_integrate(n1_for_int, n2_for_int, n3_for_int, mask_integrate.astype(np.uint8))
except Exception as e:
    print("Integration error:", e)
    z_ransac_sm = np.full((m,n), np.nan)

show_depth_views_multi(z_ransac_sm, mask, os.path.join(FIGS_DIR, "face_depth_ransac_sm_views.png"))

# Save final albedo and smoothed normals images
imsave(albedo_img_ransac, os.path.join(FIGS_DIR, "face_ransac_albedo.png"))
print("Saved figures in", FIGS_DIR)
