#import bokeh.plotting as bpl
import cv2
#import glob
#import logging
import matplotlib.pyplot as plt
#import numpy as np

from data_proc import *
from skimage import io

try:
    cv2.setNumThreads(0)
except():
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
#from caiman.utils.utils import download_demo
#from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour

# %% dataset dependent parameters
if __name__ == '__main__':
    import os

    """
    Pipeline Settings
    ===============================================================================
    gui_enable: set to enable gui for selecting input
    folder_scan_enable: set to create input_folders at runtime by searching root
        * in this case, parameter input_folders is ignored
    """
    input_folder = r'Z:\HongliWang\Miniscope\data\Fully_Ball\customEntValHere\2024_12_16\14_06_13\miniscopeDeviceName\tiff'
    output_folder = r'Z:\HongliWang\Miniscope\data\Fully_Ball\customEntValHere\2024_12_16\14_06_13\miniscopeDeviceName\registered_caiman'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    tif_name = os.path.basename(input_folder)+'_merge0.tif'
    merged_tif = os.path.join(output_folder, tif_name)

    if os.path.exists(merged_tif):
        print('--tiff files already merged')
    else:
        merge_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if ('.tif' in f)]
        merge_files = sorted(merge_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))

        # outfile = merge_tiffs(merge_files, output_folder, tifn=tif_name)
        # print('--tiff files merged--')

    fnames = merge_files  # filename to be processe

    # %% dataset dependent parameters
    fr = 20                             # imaging rate in frames per second
    decay_time = 0.4                    # length of a typical transient in seconds

    # motion correction parameters
    strides = (32, 32)          # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (16, 16)       # overlap between pathes (size of patch strides+overlaps)
    gSig_filt = (8, 8)
    max_shifts = (60,60)          # maximum allowed rigid shifts (in pixels)
    max_deviation_rigid = 8  # maximum shifts deviation allowed for patch with respect to rigid shifts
    border_nan = 'copy'

    pw_rigid = False             # flag for performing non-rigid motion correction
    shifts_opencv = True
    #use_cuda = True
    num_frames_split = 200

    # parameters for source extraction and deconvolution
    p = 1                       # order of the autoregressive system
    gnb = 2                     # number of global background components
    merge_thr = 0.85            # merging threshold, max correlation allowed
    rf = 15                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    stride_cnmf = 6             # amount of overlap between the patches in pixels
    K = 4                       # number of components per patch
    gSig = [4, 4]               # expected half size of neurons in pixels
    method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
    ssub = 1                    # spatial subsampling during initialization
    tsub = 1                    # temporal subsampling during intialization

    # parameters for component evaluation
    min_SNR = 2.0               # signal to noise ratio for accepting a component
    rval_thr = 0.85              # space correlation threshold for accepting a component
    cnn_thr = 0.99              # threshold for CNN based classifier
    cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected

    opts_dict = {'fnames': fnames,
                'fr': fr,
                'decay_time': decay_time,
                'strides': strides,
                'overlaps': overlaps,
                'max_shifts': max_shifts,
                'gSig_filt': gSig_filt,
                'border_nan': border_nan,
                'num_frames_split': num_frames_split,
                'shifts_opencv': shifts_opencv,
                'max_deviation_rigid': max_deviation_rigid,
                #'use_cuda': use_cuda,
                'pw_rigid': pw_rigid,
                'p': p,
                'nb': gnb,
                'rf': rf,
                'K': K,
                'gSig': gSig,
                'stride': stride_cnmf,
                'method_init': method_init,
                'rolling_sum': True,
                'only_init': True,
                'ssub': ssub,
                'tsub': tsub,
                'merge_thr': merge_thr,
                'min_SNR': min_SNR,
                'rval_thr': rval_thr,
                'use_cnn': True,
                'min_cnn_thr': cnn_thr,
                'cnn_lowest': cnn_lowest}

    opts = params.CNMFParams(params_dict=opts_dict)

    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

    # first we create a motion correction object with the parameters specified
    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))

    # rigid motion correction
    nIter = 3
    for i in range(nIter):
        if i==0:
            mc.motion_correct(save_movie=True)
        else:
            mc.template = mc.mmap_file  # use the template obtained before to save in computation (optional)
            mc.motion_correct(save_movie=True, template=mc.total_template_rig)

    fname_mc = mc.fname_tot_els if pw_rigid else mc.fname_tot_rig
    if pw_rigid:
        bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                     np.max(np.abs(mc.y_shifts_els)))).astype(int)
    else:
        bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(int)
        # Plot shifts
        plt.plot(mc.shifts_rig)  # % plot rigid shifts
        plt.legend(['x shifts', 'y shifts'])
        plt.xlabel('frames')
        plt.ylabel('pixels')
        plt.gcf().set_size_inches(6,3)

    bord_px = 0 if border_nan == 'copy' else bord_px
    fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
                               border_to_0=bord_px)

    # delete previous memmap files


    movie_orig = cm.load(merge_files)
    movie_corrected = cm.load(mc.mmap_file)
    #movie_corrected.save(os.path.join(output_folder, 'corrected.tif'), compress=0.2)# load motion corrected movie

    big_file = np.memmap(mc.mmap_file, mode='r', dtype=np.int16,shape=movie_corrected.shape)
    #img_tosave = np.transpose(np.reshape(big_file, [dims[1], dims[2], int(auxlen)]))
    io.imsave(os.path.join(output_folder, 'corrected.tif'), big_file, plugin='tifffile')  # saves each plane different tiff
    del big_file

    # %%capture
    # % compute metrics for the results (TAKES TIME!!)
    # %%capture
    ds_ratio = 0.2
    cm.concatenate([movie_orig.resize(1, 1, ds_ratio) - mc.min_mov * mc.nonneg_movie,
                    movie_corrected.resize(1, 1, ds_ratio)],
                   axis=2).play(fr=10,
                                gain=0.9,
                                magnification=2)


    # %%

    # pw-rigid iteratively
    mc.pw_rigid = True  # turn the flag to True for pw-rigid motion correction

    nIter = 3
    for i in range(nIter):
       mc.template = mc.mmap_file  # use the template obtained before to save in computation (optional)

       mc.motion_correct(save_movie=True, template=mc.total_template_rig)

    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    plt.plot(mc.x_shifts_els)
    plt.ylabel('x shifts (pixels)')
    plt.subplot(2, 1, 2)
    plt.plot(mc.y_shifts_els)
    plt.ylabel('y_shifts (pixels)')
    plt.xlabel('frames')

    m_els = cm.load(mc.fname_tot_els)
    downsample_ratio = 0.2
    m_els.resize(1, 1, downsample_ratio).play(
       q_max=99.5, fr=20, magnification=2)

    # %% quality assessment
    # plt.figure(figsize=(20, 10))
    # plt.subplot(1, 3, 1);
    # plt.imshow(movie_orig.local_correlations(eight_neighbours=True, swap_dim=False))
    # plt.subplot(1, 3, 2);
    # plt.imshow(m_rig.local_correlations(eight_neighbours=True, swap_dim=False))
    # plt.subplot(1, 3, 3);
    # plt.imshow(m_els.local_correlations(eight_neighbours=True, swap_dim=False))

    final_size = np.subtract(mc.total_template_els.shape, 2 * bord_px_els)  # remove pixels in the boundaries
    winsize = 100
    swap_dim = False
    resize_fact_flow = .2  # downsample for computing ROF

    tmpl_orig, correlations_orig, flows_orig, norms_orig, crispness_orig = cm.motion_correction.compute_metrics_motion_correction(
        fnames[0], final_size[0], final_size[1], swap_dim, winsize=winsize, play_flow=False,
        resize_fact_flow=resize_fact_flow)

    tmpl_rig, correlations_rig, flows_rig, norms_rig, crispness_rig = cm.motion_correction.compute_metrics_motion_correction(
        mc.fname_tot_rig[0], final_size[0], final_size[1],
        swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)

    tmpl_els, correlations_els, flows_els, norms_els, crispness_els = cm.motion_correction.compute_metrics_motion_correction(
        mc.fname_tot_els[0], final_size[0], final_size[1],
        swap_dim, winsize=winsize, play_flow=False, resize_fact_flow=resize_fact_flow)

    # %%
    plt.figure(figsize=(20, 10))
    plt.subplot(211);
    plt.plot(correlations_orig);
    plt.plot(correlations_rig);
    plt.plot(correlations_els)
    plt.legend(['Original', 'Rigid', 'PW-Rigid'])
    plt.subplot(223);
    plt.scatter(correlations_orig, correlations_rig);
    plt.xlabel('Original');
    plt.ylabel('Rigid');
    plt.plot([0.3, 0.7], [0.3, 0.7], 'r--')
    axes = plt.gca();
    axes.set_xlim([0.3, 0.7]);
    axes.set_ylim([0.3, 0.7]);
    plt.axis('square');
    plt.subplot(224);
    plt.scatter(correlations_rig, correlations_els);
    plt.xlabel('Rigid');
    plt.ylabel('PW-Rigid');
    plt.plot([0.3, 0.7], [0.3, 0.7], 'r--')
    axes = plt.gca();
    axes.set_xlim([0.3, 0.7]);
    axes.set_ylim([0.3, 0.7]);
    plt.axis('square');

    # %%
    # print crispness values
    print('Crispness original: ' + str(int(crispness_orig)))
    print('Crispness rigid: ' + str(int(crispness_rig)))
    print('Crispness elastic: ' + str(int(crispness_els)))

    # memory mappin
    # memory map the file in order 'C'
    border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
    fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                               border_to_0=border_to_0, dview=dview) # exclude borders

    # now load the file
    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
        #load frames in python format (T x X x Y)

    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)

    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)