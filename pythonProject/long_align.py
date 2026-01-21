import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.image as mpimg

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
import h5py
from scipy.ndimage import shift
import re

matplotlib.use('TkAgg')
class ImageAligner:
    def __init__(self, root_path):

        self.root_path = root_path

        def extract_trial_num(filepath):
            # Extract number after 'trial' using regex
            match = re.search(r'trial(\d+)', filepath)
            return int(match.group(1)) if match else -1  # -1 if no match

        self.dfffiles = glob.glob(os.path.join(max_projection_path, '*dff.h5'))
        self.sessions = np.full((len(self.dfffiles)), '', dtype=object)
        for ii in range(len(self.dfffiles)):
            self.sessions[ii] = os.path.basename(self.dfffiles[ii])[:-3]

        self.sessions = sorted(self.sessions,key =extract_trial_num)

        max_proj = glob.glob(os.path.join(max_projection_path, 'MAX*.tif'))
        max_proj = sorted(max_proj,key =extract_trial_num)

        max_image = []
        for ii in range(len(max_proj)):
            max_image.append(tifffile.imread(max_proj[ii]))

        img1 = max_image[0]
        img2 = max_image[1]
        self.fov_height, self.fov_width = img1.shape

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
        self.img1 = image_eq1
        self.img2 = image_eq2

        self.align_gui()
    def align_gui(self):
        self.offset = [0, 0]  # initial offset: (y, x)
        self.dragging = False
        self.prev_mouse = None

        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.img1, cmap='seismic')
        self.overlay = self.ax.imshow(self.img2, cmap='seismic', alpha=0.5, extent=(0, self.img2.shape[1], self.img2.shape[0], 0))
        self.ax.set_title("Drag the overlay (jet) image to align")

        # Buttons
        ax_button = plt.axes([0.4, 0.05, 0.2, 0.075])
        self.save_button = Button(ax_button, 'Confirm Offset')
        self.save_button.on_clicked(self.on_confirm)

        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        plt.show()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        self.dragging = True
        self.prev_mouse = (event.xdata, event.ydata)

    def on_release(self, event):
        self.dragging = False
        self.prev_mouse = None

    def on_motion(self, event):
        if not self.dragging or event.xdata is None or event.ydata is None:
            return

        dx = event.xdata - self.prev_mouse[0]
        dy = event.ydata - self.prev_mouse[1]
        self.prev_mouse = (event.xdata, event.ydata)

        self.offset[0] -= dy
        self.offset[1] -= dx

        self.update_overlay()

    def update_overlay(self):
        x0 = self.offset[1]
        y0 = self.offset[0]
        self.overlay.set_extent((x0, x0 + self.img2.shape[1],
                                 y0 + self.img2.shape[0], y0))
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key == 'enter':
            print(f"Final offset: x = {self.offset[1]:.2f}, y = {self.offset[0]:.2f}")
            plt.close(self.fig)

    def on_confirm(self, event):
        x_off, y_off = self.offset[1], self.offset[0]
        print(f"✅ Confirmed offset: x = {x_off:.2f}, y = {y_off:.2f}")

        # Save to file
        offset_dict = {"x_offset": float(x_off), "y_offset": float(y_off)}
        #with open("alignment_offset.json", "w") as f:
        #    json.dump(offset_dict, f, indent=2)
        # print("📝 Offset saved to alignment_offset.json")
        plt.close(self.fig)

    def align_rois(self):
        # find the hdf5 file with dff.hdf5
        nFiles = len(self.dfffiles)
        # load the dff
        for ii in range(nFiles):
            if ii==0:
                with h5py.File(self.dfffiles[ii], 'r') as f:
                    print("Keys (top-level datasets or groups):")
                    for key in f.keys():
                        print(" ", key)

                    # Inspect individual datasets
                    if 'dff' in f:
                        dff = f['dff'][:]
                        print(f"\ndff shape: {dff.shape}, dtype: {dff.dtype}")

                    if 'mask' in f:
                        base_mask = f['mask'][:]

            else:
                # set the ROI offset with offset x and offset y
                with h5py.File(self.dfffiles[ii], 'r') as f:
                    print("Keys (top-level datasets or groups):")
                    for key in f.keys():
                        print(" ", key)

                    # Inspect individual datasets
                    if 'dff' in f:
                        dff = f['dff'][:]
                        print(f"\ndff shape: {dff.shape}, dtype: {dff.dtype}")

                    if 'mask' in f:
                        mask = f['mask'][:]
                        print(f"\nmask shape: {mask.shape}, dtype: {mask.dtype}")

                nROIs = dff.shape[0]
                mask_reshape = mask.reshape((self.fov_height, self.fov_width,nROIs), order='F')

                shifted_roi = np.full((mask_reshape.shape), np.nan)
                for i in range(nROIs):
                    shifted_roi[:,:,i] = shift(mask_reshape[:,:,i], shift=(self.offset[0], self.offset[1]), mode='constant', cval=0,order = 0)

                fig, ax = plt.subplots(figsize=(8, 8))

                # Background image (optional)
                ax.set_title("ROI Contours: File1 (red), File2 (cyan)")
                ax.set_xlim([0, self.fov_width])
                ax.set_ylim([self.fov_height, 0])
                ax.axis('off')

                # Plot ROIs from file1 in red
                for i in range(base_mask.shape[1]):
                    roi_sparse = base_mask[:, i].reshape((self.fov_height, self.fov_width), order='F')
                    if hasattr(roi_sparse, "toarray"):
                        roi = roi_sparse.toarray()
                    else:
                        roi = roi_sparse

                    #ax.imshow(roi, cmap='Reds', alpha=0.8)

                    # Draw contour line on ROI boundary (level slightly above 0)
                    contours = ax.contour(roi, levels=[roi[roi > 0].min()], colors='red', linewidths=3)

                cyan_cmap = LinearSegmentedColormap.from_list("cyan_cmap", ["white", "cyan"])
                # Plot ROIs from file2 in cyan
                for i in range(nROIs):
                    roi_sparse = shifted_roi[:,:, i]
                    if hasattr(roi_sparse, "toarray"):
                        roi = roi_sparse.toarray()
                    else:
                        roi = roi_sparse

                    #ax.imshow(roi, cmap=cyan_cmap, alpha=0.8)

                    # Draw contour line on ROI boundary (level slightly above 0)
                    contours = ax.contour(roi, levels=[roi[roi > 0].min()], colors='cyan', linewidths=3)

                plt.tight_layout()
                plt.show()

# Example usage
if __name__ == '__main__':
    max_projection_path = r'/media/linda/WD_red_4TB/CalciumImage/Rotarod/long_align/'



    aligner = ImageAligner(max_projection_path)
    aligner.align_rois()
