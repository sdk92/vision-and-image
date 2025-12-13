#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vision and Image Processing - Assignment 3
Sections 4 & 5: shiny_vase and shiny_vase2 Datasets

Students: [Your names here]
Date: December 2024

This script implements photometric stereo analysis for:
- Section 4: shiny_vase (3 images)
- Section 5: shiny_vase2 (22 images)

Both using Woodham and RANSAC methods.
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ps_utils
import time

plt.rcParams['figure.figsize'] = (20, 14)


def process_dataset_woodham(I, S, mask, dataset_name):
    """
    Process a dataset using Woodham's method
    
    Parameters:
    -----------
    I : ndarray (m, n, k)
        k images of size (m, n)
    S : ndarray (k, 3) or (3, k)
        Light source directions
    mask : ndarray (m, n)
        Binary mask
    dataset_name : str
        Name for display
        
    Returns:
    --------
    albedo, n1, n2, n3 : ndarrays
        Estimated albedo and normal components
    """
    print(f"\n{'='*70}")
    print(f"Processing {dataset_name} - Woodham Method")
    print(f"{'='*70}")
    
    # Ensure S is (k, 3)
    if S.shape[0] == 3 and S.shape[1] != 3:
        S = S.T
    if S.shape[0] == 3:
        S = S.T
    
    m, n, k = I.shape
    print(f"Image size: {m} x {n}")
    print(f"Number of images: {k}")
    print(f"Light matrix S shape: {S.shape}")
    
    # Get mask indices
    nz = np.where(mask > 0)
    n_pixels = len(nz[0])
    print(f"Number of pixels in mask: {n_pixels}")
    
    # Create J matrix: (k, nz)
    # Each column is one pixel's intensities across k images
    J = np.zeros((k, n_pixels))
    for i in range(k):
        Ii = I[:, :, i]
        J[i, :] = Ii[nz]
    
    print(f"J matrix shape: {J.shape}")
    print(f"S matrix shape: {S.shape}")
    
    # Compute M = S^(-1) * J or S^† * J
    start_time = time.time()
    if k == 3:
        # Use inverse for 3x3 case
        try:
            S_inv = la.inv(S)
            M = S_inv @ J
            print("Using matrix inverse (3x3 case)")
        except:
            print("Warning: Matrix not invertible, using pseudo-inverse")
            S_pinv = la.pinv(S)
            M = S_pinv @ J
    else:
        # Use pseudo-inverse for overdetermined case
        S_pinv = la.pinv(S)
        M = S_pinv @ J
        print(f"Using pseudo-inverse ({k}x3 case)")
    
    print(f"M matrix shape: {M.shape}")
    
    # Extract albedo: ||M||
    Rho = la.norm(M, axis=0)
    
    # Extract normals: M / ||M||
    N = M / (Rho + 1e-10)
    
    # Reconstruct full images
    albedo = np.zeros((m, n))
    albedo[nz] = Rho
    
    n1 = np.zeros((m, n))
    n2 = np.zeros((m, n))
    n3 = np.ones((m, n))
    n1[nz] = N[0, :]
    n2[nz] = N[1, :]
    n3[nz] = N[2, :]
    
    elapsed = time.time() - start_time
    print(f"Woodham estimation completed in {elapsed:.2f} seconds")
    
    # Statistics
    print(f"\nAlbedo statistics:")
    print(f"  Mean: {np.mean(Rho):.4f}")
    print(f"  Std:  {np.std(Rho):.4f}")
    print(f"  Min:  {np.min(Rho):.4f}")
    print(f"  Max:  {np.max(Rho):.4f}")
    
    return albedo, n1, n2, n3


def process_dataset_ransac(I, S, mask, dataset_name, threshold=2.0):
    """
    Process a dataset using RANSAC
    
    Parameters:
    -----------
    I : ndarray (m, n, k)
        k images of size (m, n)
    S : ndarray (k, 3) or (3, k)
        Light source directions
    mask : ndarray (m, n)
        Binary mask
    dataset_name : str
        Name for display
    threshold : float
        RANSAC threshold
        
    Returns:
    --------
    albedo, n1, n2, n3 : ndarrays
        Estimated albedo and normal components
    """
    print(f"\n{'='*70}")
    print(f"Processing {dataset_name} - RANSAC Method (threshold={threshold})")
    print(f"{'='*70}")
    
    # Ensure S is (k, 3)
    if S.shape[0] == 3 and S.shape[1] != 3:
        S = S.T
    if S.shape[0] == 3:
        S = S.T
    
    m, n, k = I.shape
    nz = np.where(mask > 0)
    n_pixels = len(nz[0])
    
    print(f"Processing {n_pixels} pixels with RANSAC...")
    print(f"This may take several minutes for large datasets...")
    
    albedo = np.zeros((m, n))
    n1 = np.zeros((m, n))
    n2 = np.zeros((m, n))
    n3 = np.ones((m, n))
    
    start_time = time.time()
    
    # Process each pixel
    processed = 0
    failed = 0
    
    for idx in range(n_pixels):
        if processed % 5000 == 0 and processed > 0:
            elapsed = time.time() - start_time
            rate = processed / elapsed
            remaining = (n_pixels - processed) / rate
            print(f"  Progress: {processed}/{n_pixels} ({100*processed/n_pixels:.1f}%) "
                  f"- Est. time remaining: {remaining/60:.1f} min")
        
        i, j = nz[0][idx], nz[1][idx]
        
        # Get pixel intensities across all images
        pixel_intensities = I[i, j, :]
        
        # Call RANSAC
        data = (pixel_intensities, S)
        result = ps_utils.ransac_3dvector(data, threshold, verbose=0, max_iters=500)
        
        if result is not None:
            M_pixel, inliers, best_fit = result
            
            # Extract albedo and normal
            rho = la.norm(M_pixel)
            if rho > 1e-10:
                normal = M_pixel / rho
            else:
                normal = np.array([0, 0, 1])
            
            albedo[i, j] = rho
            n1[i, j] = normal[0]
            n2[i, j] = normal[1]
            n3[i, j] = normal[2]
        else:
            # RANSAC failed, use default
            n3[i, j] = 1.0
            failed += 1
        
        processed += 1
    
    elapsed = time.time() - start_time
    print(f"\n  Progress: {n_pixels}/{n_pixels} (100.0%)")
    print(f"RANSAC estimation completed in {elapsed:.2f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Failed pixels: {failed} ({100*failed/n_pixels:.2f}%)")
    
    # Statistics
    Rho = albedo[mask > 0]
    print(f"\nAlbedo statistics:")
    print(f"  Mean: {np.mean(Rho):.4f}")
    print(f"  Std:  {np.std(Rho):.4f}")
    print(f"  Min:  {np.min(Rho):.4f}")
    print(f"  Max:  {np.max(Rho):.4f}")
    
    return albedo, n1, n2, n3


def visualize_results(I, mask, dataset_name,
                     albedo_w, n1_w, n2_w, n3_w,
                     albedo_r, n1_r, n2_r, n3_r,
                     n1_s, n2_s, n3_s,
                     z_w, z_r):
    """
    Create essential visualization of results - simplified version
    """
    print(f"\nGenerating visualizations for {dataset_name}...")
    
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f'{dataset_name} - Photometric Stereo Results', fontsize=14, fontweight='bold')
    
    # Row 1: Sample images + Albedo comparison
    n_imgs = min(3, I.shape[2])
    for i in range(n_imgs):
        ax = fig.add_subplot(3, 5, i + 1)
        ax.imshow(I[:, :, i], cmap='gray')
        ax.set_title(f'Image {i+1}', fontsize=10)
        ax.axis('off')
    
    ax = fig.add_subplot(3, 5, 4)
    im = ax.imshow(albedo_w * mask, cmap='gray')
    ax.set_title('Woodham Albedo', fontsize=10, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = fig.add_subplot(3, 5, 5)
    im = ax.imshow(albedo_r * mask, cmap='gray')
    ax.set_title('RANSAC Albedo', fontsize=10, fontweight='bold')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 2: Normal fields
    ax = fig.add_subplot(3, 5, 6)
    normal_rgb_w = np.stack([n1_w, n2_w, n3_w], axis=2)
    normal_rgb_w = (normal_rgb_w + 1) / 2 * mask[:, :, np.newaxis]
    ax.imshow(normal_rgb_w)
    ax.set_title('Woodham Normals', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(3, 5, 7)
    normal_rgb_r = np.stack([n1_r, n2_r, n3_r], axis=2)
    normal_rgb_r = (normal_rgb_r + 1) / 2 * mask[:, :, np.newaxis]
    ax.imshow(normal_rgb_r)
    ax.set_title('RANSAC Normals', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    ax = fig.add_subplot(3, 5, 8)
    normal_rgb_s = np.stack([n1_s, n2_s, n3_s], axis=2)
    normal_rgb_s = (normal_rgb_s + 1) / 2 * mask[:, :, np.newaxis]
    ax.imshow(normal_rgb_s)
    ax.set_title('Smoothed Normals', fontsize=10, fontweight='bold')
    ax.axis('off')
    
    # Differences
    ax = fig.add_subplot(3, 5, 9)
    albedo_diff = np.abs(albedo_w - albedo_r) * mask
    im = ax.imshow(albedo_diff, cmap='hot')
    ax.set_title('Albedo Difference', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = fig.add_subplot(3, 5, 10)
    mask_bool = mask > 0
    dot_product = (n1_w * n1_r + n2_w * n2_r + n3_w * n3_r)
    dot_product = np.clip(dot_product, -1, 1)
    angle_diff = np.arccos(dot_product) * 180 / np.pi * mask
    im = ax.imshow(angle_diff, cmap='hot', vmin=0, vmax=60)
    ax.set_title('Normal Angle Diff (°)', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Row 3: Depth maps + 3D views
    ax = fig.add_subplot(3, 5, 11)
    z_display_w = np.nan_to_num(z_w) * mask
    im = ax.imshow(z_display_w, cmap='viridis')
    ax.set_title('Woodham Depth', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    ax = fig.add_subplot(3, 5, 12)
    z_display_r = np.nan_to_num(z_r) * mask
    im = ax.imshow(z_display_r, cmap='viridis')
    ax.set_title('RANSAC Depth', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 3D visualizations (2 views)
    for idx, (z, title, azim) in enumerate([
        (z_w, 'Woodham 3D', 45),
        (z_r, 'RANSAC 3D', 45)
    ]):
        ax = fig.add_subplot(3, 5, 13 + idx, projection='3d')
        y, x = np.mgrid[0:z.shape[0]:4, 0:z.shape[1]:4]
        z_sampled = np.nan_to_num(z[::4, ::4])
        mask_sampled = mask[::4, ::4]
        
        surf = ax.plot_surface(x, y, z_sampled * mask_sampled,
                              cmap='viridis', alpha=0.9, antialiased=True,
                              linewidth=0, rstride=1, cstride=1)
        ax.set_title(title, fontsize=10)
        ax.view_init(elev=30, azim=azim)
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.tick_params(labelsize=7)
    
    # One more 3D view from different angle
    ax = fig.add_subplot(3, 5, 15, projection='3d')
    y, x = np.mgrid[0:z_w.shape[0]:4, 0:z_w.shape[1]:4]
    z_sampled = np.nan_to_num(z_w[::4, ::4])
    mask_sampled = mask[::4, ::4]
    surf = ax.plot_surface(x, y, z_sampled * mask_sampled,
                          cmap='viridis', alpha=0.9, antialiased=True,
                          linewidth=0, rstride=1, cstride=1)
    ax.set_title('3D View (135°)', fontsize=10)
    ax.view_init(elev=30, azim=135)
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.set_zlabel('Z', fontsize=8)
    ax.tick_params(labelsize=7)
    
    plt.tight_layout()
    
    return fig


def analyze_and_report(mask, dataset_name,
                       albedo_w, n1_w, n2_w, n3_w,
                       albedo_r, n1_r, n2_r, n3_r):
    """
    Generate quantitative analysis and report
    """
    print(f"\n{'='*70}")
    print(f"Quantitative Analysis: {dataset_name}")
    print(f"{'='*70}")
    
    mask_bool = mask > 0
    
    # Albedo comparison
    albedo_diff = np.abs(albedo_w - albedo_r)[mask_bool]
    print(f"\n【Albedo Comparison】")
    print(f"  Mean difference:   {np.mean(albedo_diff):.6f}")
    print(f"  Std difference:    {np.std(albedo_diff):.6f}")
    print(f"  Median difference: {np.median(albedo_diff):.6f}")
    print(f"  Max difference:    {np.max(albedo_diff):.6f}")
    
    # Normal angle difference
    dot_product = n1_w * n1_r + n2_w * n2_r + n3_w * n3_r
    dot_product = np.clip(dot_product, -1, 1)
    angle_diff = np.arccos(dot_product[mask_bool]) * 180 / np.pi
    
    print(f"\n【Normal Vector Angle Difference】")
    print(f"  Mean angle:   {np.mean(angle_diff):.3f}°")
    print(f"  Std angle:    {np.std(angle_diff):.3f}°")
    print(f"  Median angle: {np.median(angle_diff):.3f}°")
    print(f"  Max angle:    {np.max(angle_diff):.3f}°")
    
    # Significant differences
    thresholds = [5, 10, 20, 45]
    print(f"\n【Pixels with Significant Angle Differences】")
    for thresh in thresholds:
        n_sig = np.sum(angle_diff > thresh)
        pct = 100 * n_sig / len(angle_diff)
        print(f"  > {thresh:2d}°: {n_sig:6d} pixels ({pct:5.2f}%)")
    
    # Euclidean distance
    normal_dist = np.sqrt((n1_w - n1_r)**2 + (n2_w - n2_r)**2 + (n3_w - n3_r)**2)[mask_bool]
    print(f"\n【Normal Vector Euclidean Distance】")
    print(f"  Mean distance: {np.mean(normal_dist):.6f}")
    print(f"  Max distance:  {np.max(normal_dist):.6f}")
    
    return {
        'albedo_diff_mean': np.mean(albedo_diff),
        'angle_diff_mean': np.mean(angle_diff),
        'angle_diff_max': np.max(angle_diff),
        'pct_over_20deg': 100 * np.sum(angle_diff > 20) / len(angle_diff)
    }


def section_4_shiny_vase():
    """
    Section 4: Process shiny_vase dataset (3 images)
    """
    print("\n" + "="*70)
    print("SECTION 4: SHINY VASE DATASET")
    print("="*70)
    
    # Load data
    I, mask, S = ps_utils.read_data_file('shiny_vase')
    I = I.astype(float) / 255.0 if I.dtype == np.uint8 else I.astype(float)
    mask = mask.astype(float)
    
    print(f"\nDataset loaded:")
    print(f"  Images: {I.shape}")
    print(f"  Mask: {mask.shape}")
    print(f"  Light sources: {S.shape}")
    
    # Woodham method
    albedo_w, n1_w, n2_w, n3_w = process_dataset_woodham(I, S, mask, 'shiny_vase')
    
    # Depth reconstruction (Woodham)
    print("\nIntegrating depth from Woodham normals...")
    z_w = ps_utils.unbiased_integrate(n1_w, n2_w, n3_w, mask)
    print("Depth integration complete")
    
    # RANSAC method
    albedo_r, n1_r, n2_r, n3_r = process_dataset_ransac(I, S, mask, 'shiny_vase', threshold=2.0)
    
    # Depth reconstruction (RANSAC)
    print("\nIntegrating depth from RANSAC normals...")
    z_r = ps_utils.unbiased_integrate(n1_r, n2_r, n3_r, mask)
    print("Depth integration complete")
    
    # Smooth normal field
    print("\nSmoothing RANSAC normal field...")
    n1_s, n2_s, n3_s = ps_utils.smooth_normal_field(
        n1_r, n2_r, n3_r, mask,
        iters=5, tau=0.05, verbose=True
    )
    print("Smoothing complete")
    
    # Analysis
    stats = analyze_and_report(mask, 'shiny_vase',
                               albedo_w, n1_w, n2_w, n3_w,
                               albedo_r, n1_r, n2_r, n3_r)
    
    # Visualization
    fig = visualize_results(I, mask, 'shiny_vase',
                           albedo_w, n1_w, n2_w, n3_w,
                           albedo_r, n1_r, n2_r, n3_r,
                           n1_s, n2_s, n3_s,
                           z_w, z_r)
    
    plt.savefig('section4_shiny_vase_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Results saved to: section4_shiny_vase_results.png")
    
    return stats


def section_5_shiny_vase2():
    """
    Section 5: Process shiny_vase2 dataset (22 images)
    """
    print("\n" + "="*70)
    print("SECTION 5: SHINY VASE2 DATASET")
    print("="*70)
    
    # Load data
    I, mask, S = ps_utils.read_data_file('shiny_vase2')
    I = I.astype(float)
    mask = mask.astype(float)
    
    print(f"\nDataset loaded:")
    print(f"  Images: {I.shape}")
    print(f"  Mask: {mask.shape}")
    print(f"  Light sources: {S.shape}")
    
    # Woodham method (using pseudo-inverse)
    albedo_w, n1_w, n2_w, n3_w = process_dataset_woodham(I, S, mask, 'shiny_vase2')
    
    # Depth reconstruction (Woodham)
    print("\nIntegrating depth from Woodham normals...")
    z_w = ps_utils.unbiased_integrate(n1_w, n2_w, n3_w, mask)
    print("Depth integration complete")
    
    # RANSAC method
    albedo_r, n1_r, n2_r, n3_r = process_dataset_ransac(I, S, mask, 'shiny_vase2', threshold=2.0)
    
    # Depth reconstruction (RANSAC)
    print("\nIntegrating depth from RANSAC normals...")
    z_r = ps_utils.unbiased_integrate(n1_r, n2_r, n3_r, mask)
    print("Depth integration complete")
    
    # Smooth normal field
    print("\nSmoothing RANSAC normal field...")
    n1_s, n2_s, n3_s = ps_utils.smooth_normal_field(
        n1_r, n2_r, n3_r, mask,
        iters=5, tau=0.05, verbose=True
    )
    print("Smoothing complete")
    
    # Analysis
    stats = analyze_and_report(mask, 'shiny_vase2',
                                albedo_w, n1_w, n2_w, n3_w,
                                albedo_r, n1_r, n2_r, n3_r)
    
    # Visualization
    fig = visualize_results(I, mask, 'shiny_vase2',
                            albedo_w, n1_w, n2_w, n3_w,
                            albedo_r, n1_r, n2_r, n3_r,
                            n1_s, n2_s, n3_s,
                            z_w, z_r)
    
    plt.savefig('section5_shiny_vase2_results.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Results saved to: section5_shiny_vase2_results.png")
    
    return stats


def main():
    """
    Main execution function
    """
    print("="*70)
    print("VISION AND IMAGE PROCESSING - ASSIGNMENT 3")
    print("Sections 4 & 5: Photometric Stereo with RANSAC")
    print("="*70)
    
    # Section 4
    print("\n\nExecuting Section 4...")
    stats_4 = section_4_shiny_vase()
    
    # Section 5
    print("\n\nExecuting Section 5...")
    stats_5 = section_5_shiny_vase2()
    
    # Summary
    print("\n" + "="*70)
    print("EXECUTION COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - section4_shiny_vase_results.png")
    print("  - section5_shiny_vase2_results.png (if RANSAC was run)")
    print("\nPlease review the results and include them in your report.")
    
    plt.show()


if __name__ == "__main__":
    main()