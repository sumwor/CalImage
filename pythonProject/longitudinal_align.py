# align field of view of different sessions after motion correction to allow longitudinal imaging

import os
import glob
import tifffile
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import correlate2d
from skimage import io, exposure, img_as_float
from mpl_toolkits.mplot3d import Axes3D
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from matplotlib.colors import LinearSegmentedColormap

matplotlib.use('TkAgg')

#1. max projection after mition correction test
max_projection_path = r'/media/linda/WD_red_4TB/CalciumImage/Rotarod/long_align/'

max_proj = glob.glob(os.path.join(max_projection_path, 'MAX*.tif'))
max_image = []
for ii in range(len(max_proj)):
    max_image.append(tifffile.imread(max_proj[ii]))

img1 = max_image[0]
img2 = max_image[1]

image1 = img_as_float(img1)

# Apply CLAHE (adaptive histogram equalization)
# 'clip_limit' controls contrast amplification (higher = more contrast)
if image1.min() < 0 or image1.max() > 1:
    image1 = (image1 - image1.min()) / (image1.max() - image1.min() + 1e-8)

image_eq1 = exposure.equalize_adapthist(image1, clip_limit=0.03)

image2 = img_as_float(img2)

# Apply CLAHE (adaptive histogram equalization)
# 'clip_limit' controls contrast amplification (higher = more contrast)
if image2.min() < 0 or image2.max() > 1:
    image2 = (image2 - image2.min()) / (image2.max() - image2.min() + 1e-8)

image_eq2 = exposure.equalize_adapthist(image2, clip_limit=0.03)

# Display result
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(image_eq1, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Local Contrast Enhanced (CLAHE)")
plt.imshow(image_eq2, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()

# Make sure images are 2D and same dtype
assert img1.ndim == 2 and img2.ndim == 2

# Compute 2D cross-correlation (mode='full' gives full cross-correlation)
corr = correlate2d(image_eq1, image_eq2, mode='full')

# Find the position of max correlation
y_max, x_max = np.unravel_index(np.argmax(corr), corr.shape)
print(f'Max correlation at (y, x): ({y_max}, {x_max})')

center_y = img1.shape[0] - 1
center_x = img1.shape[1] - 1

# Calculate offset relative to center
offset_y = y_max - center_y
offset_x = x_max - center_x

aligned_img2 = np.roll(img2, shift=(-offset_y, -offset_x), axis=(0,1))

#%% read the ROI masks and overlay the contour

# Example usage with your images:
plot_overlap(max_image[0], max_image[1], aligned_img2)

dff_results = glob.glob(os.path.join(max_projection_path, '*results.hdf5'))
f = load_CNMF(dff_results[0])
A1 = f.estimates.A
accepted1 = f.estimates.idx_components
fov=f.estimates.Cn
f = load_CNMF(dff_results[1])
A2 = f.estimates.A
accepted2 = f.estimates.idx_components
height, width = fov.shape

fig, ax = plt.subplots(figsize=(8, 8))

# Background image (optional)
ax.set_title("ROI Contours: File1 (red), File2 (cyan)")
ax.set_xlim([0, width])
ax.set_ylim([height, 0])
ax.axis('off')

#Plot ROIs from file1 in red
for i in accepted1:
    roi_sparse = A1[:, i].reshape((height, width), order='F')
    if hasattr(roi_sparse, "toarray"):
        roi = roi_sparse.toarray()
    else:
        roi = roi_sparse

    ax.imshow(roi, cmap='Reds', alpha=0.8)

    # Draw contour line on ROI boundary (level slightly above 0)
    contours = ax.contour(roi, levels=[roi[roi > 0].min()], colors='red', linewidths=3)


cyan_cmap = LinearSegmentedColormap.from_list("cyan_cmap", ["white", "cyan"])
# Plot ROIs from file2 in cyan
for i in accepted2:
    roi_sparse = A2[:, i].reshape((height, width), order='F')
    if hasattr(roi_sparse, "toarray"):
        roi = roi_sparse.toarray()
    else:
        roi = roi_sparse

    ax.imshow(roi, cmap=cyan_cmap, alpha=0.8)

    # Draw contour line on ROI boundary (level slightly above 0)
    contours = ax.contour(roi, levels=[roi[roi > 0].min()], colors='cyan', linewidths=3)

plt.tight_layout()
plt.show()
# Use your computed cross-correlation matrix:
#plot_correlation_surface(corr)

def plot_correlation_surface(corr):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Create grid for surface
    X = np.arange(corr.shape[1])
    Y = np.arange(corr.shape[0])
    X, Y = np.meshgrid(X, Y)

    # Plot surface
    surf = ax.plot_surface(X, Y, corr, cmap='viridis')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_title('Cross-correlation surface')
    ax.set_xlabel('X lag')
    ax.set_ylabel('Y lag')
    ax.set_zlabel('Correlation value')

    plt.show()

def plot_overlap(img1, img2, aligned_img2):
    plt.figure(figsize=(12, 5))

    # Normalize images for better visualization (optional)
    def norm(im):
        im = im.astype(np.float32)
        return (im - im.min()) / (im.max() - im.min())

    img1_n = norm(img1)
    img2_n = norm(img2)
    aligned_img2_n = norm(aligned_img2)

    # Overlap before alignment
    plt.subplot(1, 2, 1)
    plt.title('Overlap before alignment')
    plt.imshow(img1_n, cmap='Reds', alpha=0.6)
    plt.imshow(img2_n, cmap='Blues', alpha=0.4)
    plt.axis('off')

    # Overlap after alignment
    plt.subplot(1, 2, 2)
    plt.title('Overlap after alignment')
    plt.imshow(img1_n, cmap='Reds', alpha=0.6)
    plt.imshow(aligned_img2_n, cmap='Blues', alpha=0.4)
    plt.axis('off')

    plt.tight_layout()
    plt.show()