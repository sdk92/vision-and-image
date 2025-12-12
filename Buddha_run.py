import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import ps_utils
import math
import time

# ----------------- CONFIG -----------------
script_dir = os.path.dirname(os.path.abspath(__file__))
MATFILE = os.path.join(script_dir, "Buddha.mat")   
USE_RANSAC = True
RANSAC_THRESHOLD = 25.0   
RANSAC_MAX_ITERS = 2000
RANSAC_MAX_DATA_TRIES = 200
SMOOTH_ITERS = 200
EPS_N3 = 1e-6           
FIGS_DIR = "figs"
VERBOSE = True
# ------------------------------------------

os.makedirs(FIGS_DIR, exist_ok=True)

def imshow_save(img, fname, cmap='gray', vmin=None, vmax=None):
    plt.figure(figsize=(6,5))
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close()

def show_albedo_and_normals_and_save(prefix, albedo, n1, n2, n3):
    # map normals from [-1,1] to [0,1] for display
    imshow_save(albedo, os.path.join(FIGS_DIR, f"{prefix}_albedo.png"), cmap='gray')
    imshow_save((n1+1)/2, os.path.join(FIGS_DIR, f"{prefix}_n1.png"), cmap='gray')
    imshow_save((n2+1)/2, os.path.join(FIGS_DIR, f"{prefix}_n2.png"), cmap='gray')
    imshow_save((n3+1)/2, os.path.join(FIGS_DIR, f"{prefix}_n3.png"), cmap='gray')

def safe_plot_surface_and_save(Z, mask, outname, title=None, elev=30, azim=45):
    # Z: depth array (m x n), mask: same shape
    X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    Zplot = Z.copy()
    # mask outside -> NaN
    Zplot[mask == 0] = np.nan

    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Zplot, rstride=1, cstride=1, linewidth=0, antialiased=True, cmap='viridis', edgecolor='none')
    if title:
        ax.set_title(title)
    # safe zlim
    try:
        zmin = np.nanmin(Zplot)
        zmax = np.nanmax(Zplot)
        if not (np.isfinite(zmin) and np.isfinite(zmax)):
            raise ValueError("zmin/zmax not finite")
        ax.set_zlim(zmin, zmax)
    except Exception as e:
        if VERBOSE:
            print(f"[safe_plot_surface] Warning: cannot determine z-limits: {e}. Using defaults.")
        ax.set_zlim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(outname, dpi=200, bbox_inches='tight')
    plt.close()

def show_depth_views_multi(Z, mask, outname, title_prefix="depth", views=None, figsize=(18,6), cmap='viridis'):
    if views is None:
        views = [(30, 45), (60, -60), (20, 120)]

    # Prepare plot data: mask-out background
    Zplot = np.array(Z, dtype=float)
    Zplot[mask == 0] = np.nan
    mZ, nZ = Zplot.shape
    X, Y = np.meshgrid(np.arange(nZ), np.arange(mZ))

    # Create figure with len(views) subplots
    fig = plt.figure(figsize=figsize)
    for i, (elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, len(views), i, projection='3d')
        # plot surface; skip if all NaN
        if np.all(~np.isfinite(Zplot)):
            # empty subplot with warning text
            ax.text(0.5, 0.5, 0.5, "No valid depth", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_zticks([])
            ax.set_title(f"{title_prefix} (elev={elev},azim={azim})")
            continue
        surf = ax.plot_surface(X, Y, Zplot, rstride=1, cstride=1, linewidth=0, antialiased=True, cmap=cmap, edgecolor='none')
        ax.view_init(elev=elev, azim=azim)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{title_prefix} (elev={elev},azim={azim})")
        # safe zlim
        try:
            zmin = np.nanmin(Zplot)
            zmax = np.nanmax(Zplot)
            if np.isfinite(zmin) and np.isfinite(zmax):
                ax.set_zlim(zmin, zmax)
        except Exception:
            pass
    plt.tight_layout()
    plt.savefig(outname, dpi=200, bbox_inches='tight')
    plt.close(fig)

# ----------------- Load data -----------------
print("Loading data from:", MATFILE)
if not os.path.exists(MATFILE):
    raise FileNotFoundError(f"Mat file not found: {MATFILE}. Put it in the script directory or change MATFILE path.")

I, mask, S = ps_utils.read_data_file(MATFILE)
m, n, nb = I.shape
print(f"Image shape: {m} x {n}  #images: {nb}")
print("S shape:", S.shape)

# mask indices
mask_idx = np.where(mask > 0)
nz = mask_idx[0].size
print("Number of mask pixels (nz):", nz)

# build J: nb x nz
J = np.zeros((nb, nz), dtype=float)
for k in range(nb):
    frame = I[:, :, k]
    J[k, :] = frame[mask_idx]

# ----------------- Woodham pinv solution -----------------
Spinv = np.linalg.pinv(S)    # shape (3, nb)
M_pinv = Spinv @ J           # 3 x nz

albedo_pinv = np.linalg.norm(M_pinv, axis=0)   # length nz
eps = 1e-8
norms = albedo_pinv.copy()
norms[norms < eps] = 1.0
N_pinv = M_pinv / norms     # 3 x nz

# reconstruct images
albedo_img_pinv = np.zeros((m, n), dtype=float)
n1_pinv = np.zeros((m, n), dtype=float)
n2_pinv = np.zeros((m, n), dtype=float)
n3_pinv = np.zeros((m, n), dtype=float)

albedo_img_pinv[mask_idx] = albedo_pinv
n1_pinv[mask_idx] = N_pinv[0, :]
n2_pinv[mask_idx] = N_pinv[1, :]
n3_pinv[mask_idx] = N_pinv[2, :]

# integrate pinv normals (use safe mask later if needed)
try:
    z_pinv = ps_utils.simchony_integrate(n1_pinv, n2_pinv, n3_pinv, mask)
except Exception as e:
    print("[pinv integrate] Exception:", e)
    z_pinv = np.full_like(mask, np.nan, dtype=float)

# save pinv results
show_albedo_and_normals_and_save("pinv", albedo_img_pinv, n1_pinv, n2_pinv, n3_pinv)
safe_plot_surface_and_save(z_pinv, mask, os.path.join(FIGS_DIR, "depth_pinv_view1.png"), title="depth pinv view1")

# ----------------- RANSAC per-pixel estimation -----------------
M_ransac = np.zeros_like(M_pinv)   # 3 x nz
inliers_count = np.zeros(nz, dtype=int)

if USE_RANSAC:
    print("Running RANSAC per pixel (this may take some time)...")
    t0 = time.time()
    for i in range(nz):
        Ivec = J[:, i]  # length nb
        data = (Ivec, S)
        res = None
        try:
            res = ps_utils.ransac_3dvector(data, threshold=RANSAC_THRESHOLD,
                                           max_data_tries=RANSAC_MAX_DATA_TRIES,
                                           max_iters=RANSAC_MAX_ITERS,
                                           p=0.99, det_threshold=1e-4, verbose=0)
        except Exception as e:
            # RANSAC internal error: print occasionally
            if VERBOSE and (i % 5000 == 0):
                print(f"[RANSAC] pixel {i} exception: {e}")
            res = None

        if res is None:
            # fallback to pinv for this pixel
            m_est = M_pinv[:, i]
            inliers = None
        else:
            m_est, inliers, fit = res
            if inliers is not None:
                inliers_count[i] = len(inliers)
        M_ransac[:, i] = m_est
    t1 = time.time()
    print(f"RANSAC finished in {t1-t0:.1f}s")
else:
    M_ransac = M_pinv.copy()

# compute albedo & normals for RANSAC
albedo_ransac = np.linalg.norm(M_ransac, axis=0)
norms = albedo_ransac.copy()
norms[norms < eps] = 1.0
N_ransac = M_ransac / norms

# reconstruct images
albedo_img_ransac = np.zeros((m, n), dtype=float)
n1_ransac = np.zeros((m, n), dtype=float)
n2_ransac = np.zeros((m, n), dtype=float)
n3_ransac = np.zeros((m, n), dtype=float)

albedo_img_ransac[mask_idx] = albedo_ransac
n1_ransac[mask_idx] = N_ransac[0, :]
n2_ransac[mask_idx] = N_ransac[1, :]
n3_ransac[mask_idx] = N_ransac[2, :]

# diagnostics before smoothing
def stats_normals(n1, n2, n3, mask, name=""):
    mask_valid = (mask > 0)
    nz = np.sum(mask_valid)
    n3_zero = np.sum(mask_valid & (np.abs(n3) < EPS_N3))
    nan_count = np.sum(np.isnan(n1[mask_valid]) | np.isnan(n2[mask_valid]) | np.isnan(n3[mask_valid]))
    inf_count = np.sum(np.isinf(n1[mask_valid]) | np.isinf(n2[mask_valid]) | np.isinf(n3[mask_valid]))
    print(f"[{name}] mask pixels: {nz}, n3 nearly zero (<{EPS_N3}): {n3_zero}, NaNs: {nan_count}, Infs: {inf_count}")

stats_normals(n1_pinv, n2_pinv, n3_pinv, mask, "pinv")
stats_normals(n1_ransac, n2_ransac, n3_ransac, mask, "ransac (before smooth)")

# inliers stats
if USE_RANSAC:
    print("RANSAC inliers stats (min, median, mean, max):",
          np.min(inliers_count), np.median(inliers_count), np.mean(inliers_count), np.max(inliers_count))
    print("Fraction pixels with >0 inliers:", np.mean(inliers_count > 0))

# ----------------- Safe integration for RANSAC normals -----------------
# Create mask_integrate where n3 is not too small and mask>0
mask_integrate_ransac = (mask > 0) & (np.abs(n3_ransac) > EPS_N3)
valid_pixels = np.sum(mask_integrate_ransac)
print("Valid pixels for integration (ransac):", valid_pixels, "/", nz)

if valid_pixels == 0:
    print("No valid pixels for RANSAC integration. Skipping integration and smoothing.")
    z_ransac = np.full_like(mask, np.nan, dtype=float)
    n1_ransac_safe = n1_ransac.copy(); n2_ransac_safe = n2_ransac.copy(); n3_ransac_safe = n3_ransac.copy()
    n1_ransac_safe[~mask] = np.nan; n2_ransac_safe[~mask] = np.nan; n3_ransac_safe[~mask] = np.nan
    n1_sm = n1_ransac_safe.copy(); n2_sm = n2_ransac_safe.copy(); n3_sm = n3_ransac_safe.copy()
else:
    # Prepare arrays to feed integration / smoothing: set outside pixels to safe defaults
    n1_for_int = n1_ransac.copy(); n2_for_int = n2_ransac.copy(); n3_for_int = n3_ransac.copy()
    # Outside mask_integrate -> assign safe values (0,0,1)
    n1_for_int[~mask_integrate_ransac] = 0.0
    n2_for_int[~mask_integrate_ransac] = 0.0
    n3_for_int[~mask_integrate_ransac] = 1.0

    # integrate
    try:
        z_ransac = ps_utils.simchony_integrate(n1_for_int, n2_for_int, n3_for_int, mask_integrate_ransac.astype(np.uint8))
    except Exception as e:
        print("[simchony_integrate error for ransac] ", e)
        z_ransac = np.full_like(mask, np.nan, dtype=float)

    # smooth normal field on valid region
    # Prepare arrays similarly for smooth function
    n1_tmp = n1_for_int.copy(); n2_tmp = n2_for_int.copy(); n3_tmp = n3_for_int.copy()
    try:
        n1_sm, n2_sm, n3_sm = ps_utils.smooth_normal_field(n1_tmp, n2_tmp, n3_tmp, mask_integrate_ransac.astype(np.uint8), iters=SMOOTH_ITERS, tau=0.05, verbose=False)
    except Exception as e:
        print("[smooth_normal_field] Exception:", e)
        # fallback: no smoothing
        n1_sm, n2_sm, n3_sm = n1_ransac.copy(), n2_ransac.copy(), n3_ransac.copy()

    # After smoothing, set outside region to NaN for visualization
    n1_sm[~mask_integrate_ransac] = np.nan
    n2_sm[~mask_integrate_ransac] = np.nan
    n3_sm[~mask_integrate_ransac] = np.nan

    # integrate smoothed normals (use safe arrays)
    n1_for_int2 = n1_sm.copy(); n2_for_int2 = n2_sm.copy(); n3_for_int2 = n3_sm.copy()
    # replace NaN outside -> safe values
    n1_for_int2[np.isnan(n1_for_int2)] = 0.0
    n2_for_int2[np.isnan(n2_for_int2)] = 0.0
    n3_for_int2[np.isnan(n3_for_int2)] = 1.0
    try:
        z_ransac_sm = ps_utils.simchony_integrate(n1_for_int2, n2_for_int2, n3_for_int2, mask_integrate_ransac.astype(np.uint8))
    except Exception as e:
        print("[simchony_integrate error for ransac smoothed] ", e)
        z_ransac_sm = np.full_like(mask, np.nan, dtype=float)

# Save RANSAC results
show_albedo_and_normals_and_save("ransac", albedo_img_ransac, n1_ransac, n2_ransac, n3_ransac)
show_albedo_and_normals_and_save("ransac_smoothed", albedo_img_ransac, n1_sm, n2_sm, n3_sm)

# Save depth views (safe plotting)
safe_plot_surface_and_save(z_ransac, mask, os.path.join(FIGS_DIR, "depth_ransac_view1.png"), title="depth ransac view1")
safe_plot_surface_and_save(z_ransac_sm, mask, os.path.join(FIGS_DIR, "depth_ransac_sm_view1.png"), title="depth ransac smoothed view1")

# Print final diagnostics
print("Finished. Figures saved in:", FIGS_DIR)
if USE_RANSAC:
    print("RANSAC: mean inliers per pixel (only >0):",
          np.mean(inliers_count[inliers_count>0]) if np.any(inliers_count>0) else 0)
    print("RANSAC: fraction pixels with >0 inliers:", np.mean(inliers_count>0))
# Count NaNs in depth outputs
print("z_pinv: NaNs count:", np.sum(~np.isfinite(z_pinv)))
print("z_ransac: NaNs count:", np.sum(~np.isfinite(z_ransac)))
print("z_ransac_sm: NaNs count:", np.sum(~np.isfinite(z_ransac_sm)))
# multi-view single-image outputs
show_depth_views_multi(z_pinv, mask, os.path.join(FIGS_DIR, "depth_pinv_views.png"), title_prefix="depth pinv")
show_depth_views_multi(z_ransac, mask, os.path.join(FIGS_DIR, "depth_ransac_views.png"), title_prefix="depth ransac")
show_depth_views_multi(z_ransac_sm, mask, os.path.join(FIGS_DIR, "depth_ransac_sm_views.png"), title_prefix="depth ransac smoothed")
