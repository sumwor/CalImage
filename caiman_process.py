

import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf.utilities import detrend_df_f
from caiman.components_evaluation import estimate_components_quality_auto
import numpy as np
import h5py, os, time
import warnings
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

# %matplotlib inline

# ignore warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

def caiman_main(fr, fnames, out, K=25, rf=25, stride_cnmf=10, z=0, dend=False):
    # modified from https://github.com/senis000/CaBMI_analysis
    """
    Main function to compute the caiman algorithm. For more details see github and papers
    fpath(str): Folder where to store the plots
    fr(int): framerate
    fnames(list-str): list with the names of the files to be computed together
    z(array): vector with the values of z relative to y
    dend(bool): Boleean to change parameters to look for neurons or dendrites
    display_images(bool): to display and save different plots
    returns
    F_dff(array): array with the dff of the components
    com(array): matrix with the position values of the components as given by caiman
    cnm(struct): struct with different stimates and returns from caiman"""
    print("K", K, "rf", rf, "stride_cnmf", stride_cnmf)

    # parameters
    decay_time = 1.25  # length of a typical transient in seconds

    # Look for the best parameters for this 2p system and never change them again :)
    # motion correction parameters
    niter_rig = 1  # number of iterations for rigid motion correction
    max_shifts = (32, 32)  # maximum allow rigid shift
    splits_rig = 10  # for parallelization split the movies in  num_splits chuncks across time
    strides = (48, 48)  # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)  # overlap between patches (size of patch strides+overlaps)
    splits_els = 10  # for parallelization split the movies in  num_splits chuncks across time
    upsample_factor_grid = 4  # upsample factor to avoid smearing when merging patches
    max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts

    # parameters for source extraction and deconvolution
    p = 1  # order of the autoregressive system
    gnb = 2  # number of global background components
    merge_thresh = 0.8  # merging threshold, max correlation allowed
    # rf = 25  # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
    # stride_cnmf = 10  # amount of overlap between the patches in pixels
    # K = 25  # number of components per patch

    if dend:
        gSig = [1, 1]  # expected half size of neurons
        init_method = 'sparse_nmf'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
        alpha_snmf = 1e-6  # sparsity penalty for dendritic data analysis through sparse NMF
    else:
        gSig = [4, 4]  # expected half size of neurons
        init_method = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
        alpha_snmf = 100  # sparsity penalty for dendritic data analysis through sparse NMF

    # parameters for component evaluation
    min_SNR = 2.5  # signal to noise ratio for accepting a component
    rval_thr = 0.8  # space correlation threshold for accepting a component
    cnn_thr = 0.8  # threshold for CNN based classifier

    downsample_ratio = .2
    dview = None  # parallel processing keeps crashing.

    print('***************Starting motion correction*************')
    print('files:')
    print(fnames)

    # %% start a cluster for parallel processing
    # %% start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)
    # c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

    # %%% MOTION CORRECTION
    # first we create a motion correction object with the parameters specified
    # min_mov = cm.load(fnames[0]).min()
    # this will be subtracted from the movie to make it non-negative

    # check if the computer has gpu, install cuda first

    # delete min_mov // HW
    mc = MotionCorrect(fnames,
                       dview=dview, max_shifts=max_shifts, niter_rig=niter_rig,
                       splits_rig=splits_rig,
                       strides=strides, overlaps=overlaps, splits_els=splits_els,
                       upsample_factor_grid=upsample_factor_grid,
                       max_deviation_rigid=max_deviation_rigid,
                       shifts_opencv=True, nonneg_movie=True)
    # note that the file is not loaded in memory

    # %% run rigid motion correction first to obtain the template
    mc.motion_correct(save_movie=True)

    # compare with original movie
    # load motion corrected movie
    m_rig = cm.load(mc.mmap_file)
    bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(int)
    # visualize templates
    # plt.figure(figsize=(20, 10))
    # plt.imshow(mc.total_template_rig, cmap='gray')
    # plt.close()
    #plt.show()

    # %% plot rigid shifts

    plt.figure(figsize=(20, 10))
    plt.plot(mc.shifts_rig)
    plt.legend(['x shifts', 'y shifts'])
    plt.xlabel('frames')
    plt.ylabel('pixels')

    # %% motion correct piecewise rigid
    mc.pw_rigid = True  # turn the flag to True for pw-rigid motion correction
    mc.template = mc.mmap_file  # use the template obtained before to save in computation (optional)

    mc.motion_correct(save_movie=True, template=mc.total_template_rig)
    m_els = cm.load(mc.fname_tot_els)
    m_els.resize(1, 1, downsample_ratio).play(
        q_max=99.5, fr=30, magnification=2, bord_px=bord_px_rig)

    m_orig = cm.load_movie_chain(fnames)
    cm.concatenate([m_orig.resize(1, 1, downsample_ratio) - mc.min_mov * mc.nonneg_movie,
                    m_rig.resize(1, 1, downsample_ratio), m_els.resize(
            1, 1, downsample_ratio)], axis=2).play(fr=60, q_max=99.5, magnification=2, bord_px=bord_px_rig)

    # %% visualize elastic shifts
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    plt.plot(mc.x_shifts_els)
    plt.ylabel('x shifts (pixels)')
    plt.subplot(2, 1, 2)
    plt.plot(mc.y_shifts_els)
    plt.ylabel('y_shifts (pixels)')
    plt.xlabel('frames')
    # %% compute borders to exclude
    bord_px_els = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)),
                                     np.max(np.abs(mc.y_shifts_els)))).astype(int)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1);
    plt.imshow(m_orig.local_correlations(eight_neighbours=True, swap_dim=False))
    plt.subplot(1, 3, 2);
    plt.imshow(m_rig.local_correlations(eight_neighbours=True, swap_dim=False))
    plt.subplot(1, 3, 3);
    plt.imshow(m_els.local_correlations(eight_neighbours=True, swap_dim=False))

# %% quality assessment
    # % compute metrics for the results (TAKES TIME!!)
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
    # print crispness values
    print('Crispness original: ' + str(int(crispness_orig)))
    print('Crispness rigid: ' + str(int(crispness_rig)))
    print('Crispness elastic: ' + str(int(crispness_els)))

    # %% plot the results of Residual Optical Flox


    totdes = [np.nansum(mc.x_shifts_els), np.nansum(mc.y_shifts_els)]

    m_els = cm.load(mc.fname_tot_els)
    border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
    # maximum shift to be used for trimming against NaNs

    print('***************Motion correction has ended*************')
    # maximum shift to be used for trimming against NaNs

    # %% MEMORY MAPPING
    # memory map the file in order 'C'
    fnames = mc.fname_tot_els  # name of the pw-rigidly corrected file.
    fname_new = cm.save_memmap(fnames, base_name='memmap_', order='C',
                               border_to_0=bord_px_els)  # exclude borders
    print(fname_new)

    # now load the file
    Yr, dims, T = cm.load_memmap(fname_new)
    d1, d2 = dims
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    # load frames in python format (T x X x Y)

    # %% restart cluster to clean up memory
    # cm.stop_server(dview=dview)
    # c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

    # %% RUN CNMF ON PATCHES
    print('***************Running CNMF...*************')

    # First extract spatial and temporal components on patches and combine them
    # for this step deconvolution is turned off (p=0)

    cnm = cnmf.CNMF(n_processes=1, k=K, gSig=gSig, merge_thresh=merge_thresh,
                    p=0, dview=dview, rf=rf, stride=stride_cnmf, memory_fact=1,
                    method_init=init_method, alpha_snmf=alpha_snmf,
                    only_init_patch=False, gnb=gnb, border_pix=bord_px_els)
    cnm = cnm.fit(images)

    # %% COMPONENT EVALUATION
    # the components are evaluated in three ways:
    #   a) the shape of each component must be correlated with the data
    #   b) a minimum peak SNR is required over the length of a transient
    #   c) each shape passes a CNN based classifier

    idx_components, idx_components_bad, SNR_comp, r_values, cnn_preds = \
        estimate_components_quality_auto(images, cnm.estimates.A, cnm.estimates.C, cnm.estimates.b,
                                         cnm.estimates.f,
                                         cnm.estimates.YrA, fr, decay_time, gSig, dims,
                                         dview=dview, min_SNR=min_SNR,
                                         r_values_min=rval_thr, use_cnn=False,
                                         thresh_cnn_min=cnn_thr)


    # %% RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
    A_in, C_in, b_in, f_in = cnm.estimates.A[:, idx_components], cnm.estimates.C[
        idx_components], cnm.estimates.b, cnm.estimates.f
    cnm2 = cnmf.CNMF(n_processes=1, k=A_in.shape[-1], gSig=gSig, p=p, dview=dview,
                     merge_thresh=merge_thresh, Ain=A_in, Cin=C_in, b_in=b_in,
                     f_in=f_in, rf=None, stride=None, gnb=gnb,
                     method_deconvolution='oasis', check_nan=True)

    print('***************Fit*************')
    cnm2 = cnm2.fit(images)

    print('***************Extracting DFFs*************')
    # %% Extract DF/F values

    # cm.stop_server(dview=dview)
    try:
        F_dff = detrend_df_f(cnm2.estimates.A, cnm2.estimates.b, cnm2.estimates.C, cnm2.estimates.f,
                             YrA=cnm2.estimates.YrA, quantileMin=8, frames_window=250)
        # F_dff = detrend_df_f(cnm.A, cnm.b, cnm.C, cnm.f, YrA=cnm.YrA, quantileMin=8, frames_window=250)
    except:
        F_dff = cnm2.estimates.C * np.nan
        print('WHAAT went wrong again?')

    print('***************stopping cluster*************')
    # %% STOP CLUSTER and clean up log files
    # cm.stop_server(dview=dview)

    # ***************************************************************************************
    # Preparing output data
    # F_dff  -> DFF values,  is a matrix [number of neurons, length recording]

    # com  --> center of mass,  is a matrix [number of neurons, 2]
    print('***************preparing output data*************')
    del fname_new
    cnm2.save(out)
    with h5py.File(out, mode='a') as fp:
        fp.create_dataset('dff', data=F_dff)
        fp.create_dataset('snr', data=SNR_comp[idx_components])

if __name__ == '__main__':
    root = "/Users/albertqu/Documents/2.Courses/CogSci127/proj/data/"  # DATA ROOT
    nodecay = root+"merge_nodecay.tif"
    fr=4
    while not os.path.exists(nodecay):
        time.sleep(1)
    print('found nodecay')
    caiman_main(fr, fnames=[nodecay],out=os.path.join(root, 'out_nodecay.hdf5'))