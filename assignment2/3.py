import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_laplace
from skimage import color
from PIL import Image

# Load the uploaded image
image_path = './mandrill.jpg'
image_color = np.array(Image.open(image_path))

# Convert to grayscale
image = color.rgb2gray(image_color)
image = (image * 255).astype(float)  # Scale to 0-255 range

print(f"Image loaded: {image.shape}")
print(f"Original color image: {image_color.shape}")

# Define sigma values
sigmas = [1, 2, 4, 8]

# define the graph
fig3, axes3 = plt.subplots(3, 3, figsize=(18, 18))
fig3.suptitle('Laplacian of Gaussian (LoG) Filtering (σ = 1, 2, 4, 8)', 
              fontsize=18, fontweight='bold')

axes3[0, 0].imshow(image, cmap='gray')
axes3[0, 0].set_title('Grayscale Image', fontsize=14, fontweight='bold')
axes3[0, 0].axis('off')

axes3[0, 1].axis('off')
axes3[0, 2].axis('off')

log_results = []

for idx, sigma in enumerate(sigmas):
    # Laplacian-Gaussian filtering in terms of different sigma
    log_image = gaussian_laplace(image, sigma=sigma)
    log_results.append(log_image)
    
    row = 1 + idx // 2
    col = idx % 2
    
    im = axes3[row, col].imshow(log_image, cmap='RdBu_r', vmin=-50, vmax=50)
    axes3[row, col].set_title(f'LoG: σ = {sigma}', fontsize=14, fontweight='bold')
    axes3[row, col].axis('off')
    plt.colorbar(im, ax=axes3[row, col], fraction=0.046)
    
    min_log = np.min(log_image)
    max_log = np.max(log_image)
    axes3[row, col].text(0.02, 0.98, f'Min: {min_log:.1f}\nMax: {max_log:.1f}', 
                        transform=axes3[row, col].transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=11)

axes3[2, 2].axis('off')

plt.tight_layout()
plt.savefig('./mandrill_log_filtering.png', dpi=300, bbox_inches='tight')
print("LoG filtering saved!")