# batch process for caiman motion correction 
import multiprocessing as mp
import cv2
import glob
import holoviews as hv
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import re
import gc
import time

import caiman as cm
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.cnmf import load_CNMF
from caiman.utils.utils import download_demo
from caiman.utils.visualization import inspect_correlation_pnr, nb_inspect_correlation_pnr
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import params as params
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
from caiman.utils.visualization import view_quilt

from utils_HW import *
from tqdm import tqdm
import tifffile

try:
    cv2.setNumThreads(0)
except:
    pass

import matplotlib
matplotlib.use("Agg")


def frame_row_corr_worker(args):
    ff, mmap_file, shape, dtype, ref = args
    images = np.memmap(mmap_file, dtype=dtype, mode='r', shape=shape)
    return frame_row_corr(images[ff], ref)

def main(root_path, if_batch=False):

    # find all sessions
    # find all image videos
    if if_batch: # if batch processing, get all folders under root_path
        session_list = glob.glob(os.path.join(root_path, '**'))
    else: # single folder processing
        session_list = [root_path]

    sessions_todo = []
    ImgVideos_list = []
    ImgName_list = []
    
    for ses in session_list:
        # check if img video exists, and processed video does not exist
        Imgvideo = glob.glob(os.path.join(ses, '*_ImgVideo*.avi'))
        if len(Imgvideo)>0:
            ses_video_list = []   # sublist for this session
            ses_name_list = []  
            sesName = os.path.basename(Imgvideo[0])[:-4] 
            if len(Imgvideo)==1:                
                processed_file = glob.glob(os.path.join(ses, 'temp', sesName+'*_C_frames_*.mmap'))
            else:
                processed_file = glob.glob(os.path.join(ses, 'temp', sesName[0]+'_combined_*_C_frames_*.mmap'))  # if multiple videos, just process all
            # if more than 1 videos, concatenate them first
            if len(processed_file)==0:
                for ii in range(len(Imgvideo)):
                    # get the file name
                    sesName = os.path.basename(Imgvideo[ii])[:-4] 
                    ses_video_list.append(Imgvideo[ii])
                    ses_name_list.append(sesName)
                ImgName_list.append(ses_name_list)
                sessions_todo.append(ses)
                ImgVideos_list.append(ses_video_list)
    
    #%% parallel computing setup
    print(f"You have {psutil.cpu_count()} CPUs available in your current environment")
    num_processors_to_use = None

    #%% start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
    if 'cluster' in locals():  # 'locals' contains list of current local variables
        print('Closing previous cluster')
        cm.stop_server(dview=cluster)
    print("Setting up new cluster")
    _, cluster, n_processes = cm.cluster.setup_cluster(backend='multiprocessing', 
                                                    n_processes=num_processors_to_use, 
                                                    ignore_preexisting=False)
    print(f"Successfully set up cluster with {n_processes} processes")

    #%% go through each session
    for idx,ses in enumerate(sessions_todo):
        
        #%% ---------------- LOGGING SETUP ----------------
        #log_dir = r'/global/scratch/users/hongliwang/miniscope/test'
        log_dir = os.path.join(ses,'temp')
        os.makedirs(log_dir, exist_ok=True)
        logfile = os.path.join(log_dir, 'preprocess.log')

        logger = logging.getLogger('preprocess')
        logger.setLevel(logging.INFO)

        logfmt = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(process)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        fh = logging.FileHandler(logfile)
        fh.setFormatter(logfmt)
        logger.addHandler(fh)
        logger.info("===== preprocess started =====")
        t_total_start = time.perf_counter()

        # limit threading
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

        os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


        movie_path = ImgVideos_list[idx]
        out_path = os.path.join(ses,'temp')
        os.makedirs(out_path, exist_ok=True)
        print('Processing session: {}'.format(ses))
        sesName = ImgName_list[idx][0][0:-9]
        # define parameters
        os.environ['CAIMAN_DATA']=ses
        # dataset dependent parameters
        frate = 20                      # movie frame rate
        decay_time = 0.3                 # length of a typical transient in seconds

        # motion correction parameters
        motion_correct = True    # flag for performing motion correction
        pw_rigid = False         # flag for performing piecewise-rigid motion correction (otherwise just rigid)
        gSig_filt = (10, 10)       # sigma for high pass spatial filter applied before motion correction, used in 1p data1
        max_shifts = (20, 20)      # maximum allowed rigid shift
        strides = (48, 48)       # start a new patch for pw-rigid motion correction every x pixels
        overlaps = (24, 24)      # overlap between patches (size of patch = strides + overlaps)
        max_deviation_rigid = 3  # maximum deviation allowed for patch with respect to rigid shifts
        border_nan = 'copy'      # replicate values along the boundaries

        mc_dict = {
            'fnames': movie_path,
            'fr': frate,
            'decay_time': decay_time,
            'pw_rigid': pw_rigid,
            'max_shifts': max_shifts,
            'gSig_filt': gSig_filt,
            'strides': strides,
            'overlaps': overlaps,
            'max_deviation_rigid': max_deviation_rigid,
            'border_nan': border_nan
        }

        parameters = params.CNMFParams(params_dict=mc_dict)
        mot_correct = MotionCorrect(movie_path, dview=cluster, **parameters.get_group('motion'))
        
        t0 = time.perf_counter() # motion correction start time
        logger.info(f"Motion correction parameters: {mc_dict}")


        if motion_correct and not os.path.exists(os.path.join(out_path, sesName + '_rigid_shifts.png')):
            # do motion correction rigid
            
            mot_correct.motion_correct(save_movie=True)
            fname_mc = mot_correct.fname_tot_els if pw_rigid else mot_correct.fname_tot_rig
            if pw_rigid:
                bord_px = np.ceil(np.maximum(np.max(np.abs(mot_correct.x_shifts_els)),
                                            np.max(np.abs(mot_correct.y_shifts_els)))).astype(int)
            else:
                bord_px = np.ceil(np.max(np.abs(mot_correct.shifts_rig))).astype(int)
                # Plot shifts
                plt.plot(mot_correct.shifts_rig)  # % plot rigid shifts
                plt.legend(['x shifts', 'y shifts'])
                plt.xlabel('frames')
                plt.ylabel('pixels')
                plt.gcf().set_size_inches(6,3)
                plt.savefig(os.path.join(out_path, sesName + '_rigid_shifts.png'))

            bord_px = 0 if border_nan == 'copy' else bord_px
            #fname_new = cm.save_memmap(fname_mc, base_name='memmap_', order='C',
            #                           border_to_0=bord_px)
        else:  # if no motion correction just memory map the file
            fname_mc = glob.glob(os.path.join(out_path, sesName+'*_rig_*_order_F*.mmap'))
            
        print('Finished processing session: {}'.format(ses))
        logger.info(f"Motion correcttion finished in ({time.perf_counter() - t0:.2f} s)")

        #%% after motion correction, find the memmap file in F order and save it in C order and tiffs

        t0 = time.perf_counter() # time for saving tiffs and calculate row correlation
        logger.info(f"Saving motion corrected tiffs and calculating row correlation started.")

        print('Saving motion corrected video in tiffs...')
        output_tiff_dir = os.path.join(ses, 'motion_corrected_tiffs') 
        tiff_files = glob.glob(os.path.join(output_tiff_dir, '*.tif')) + \
             glob.glob(os.path.join(output_tiff_dir, '*.tiff'))

        # write tiff files, also compute row_correlation to check for screen tearing simultaneously
        ave_row_corr = []
        if not len(tiff_files) > 20: # some arbitrary number
        # load memmap (NO RAM copy)
            prev_nTiff = 0  # keep track of previous number of tiffs
            for mIdx, mmap_file in enumerate(fname_mc):

                Yr, dims, T = cm.load_memmap(mmap_file)
                images = Yr.T.reshape((T,) + dims, order='F')

                shape = (T,) + dims
                dtype = Yr.dtype
                mmap_file = fname_mc[mIdx]
                nFrames = images.shape[0]
                d1, d2 = dims[:2]
                # CHANGE THIS
                frames_per_tiff = 1000
                n_tiffs = nFrames//frames_per_tiff+1
                dtype_out = np.uint16

                os.makedirs(output_tiff_dir, exist_ok=True)

                # define reference frame for row correlation
                if mIdx == 0:
                    Nref = min(1000, nFrames)
                    ref = images[:Nref].mean(axis=0).astype(np.float32)

                    # precompute reference normalization (rows 1:)
                    ref0 = ref[1:].copy()
                    ref0 -= ref0.mean(axis=1, keepdims=True)
                    ref_norm = np.sqrt(np.sum(ref0 * ref0, axis=1))
            # Load memmap (disk-backed)
            

                for i in tqdm(range(n_tiffs)):
                    start = int(i * frames_per_tiff)
                    end = int(min((i + 1) * frames_per_tiff, T))

                    if start >= T:
                        break

                    out_path_tt = os.path.join(
                        output_tiff_dir,
                        sesName + f'_motion_corrected_part_{(i+prev_nTiff):03d}.tif'
                    )
                    

                    with tifffile.TiffWriter(out_path_tt, bigtiff=True) as tif:
                        # iterate in smaller chunks inside each TIFF
                        for s in range(start, end, 1000):
                            e = min(s + 1000, end)

                            # memmap slice (NO full RAM copy)
                            Y_chunk = Yr[:, s:e]

                            # reshape -> (frames, height, width)
                            Y_chunk = Y_chunk.reshape(
                                (d1, d2, e - s),
                                order='F'
                            ).transpose(2, 0, 1)

                            corr_vals = frame_row_corr_batch(Y_chunk, ref0, ref_norm)
                            ave_row_corr.extend(corr_vals.tolist())

                            tif.write(
                                Y_chunk.astype(dtype_out, copy=False),
                                photometric='minisblack'
                            )

                prev_nTiff = prev_nTiff + n_tiffs

        # plot ave_row_corr
        savefigpath = os.path.join(out_path, sesName+'_ave_row_corr.png')
        plt.figure()
        plt.plot(ave_row_corr)
        plt.show()
        plt.savefig(savefigpath)
        plt.close()

        # check if there is screen tearing (ave_row_corr below 0.85)
        if np.any(np.array(ave_row_corr) < 0.85):
            # save the frames with low row correlation in csv
            low_corr_indices = np.where(np.array(ave_row_corr) < 0.85)[0]
            np.savetxt(os.path.join(out_path, sesName+'_low_row_corr_frames.csv'), low_corr_indices, delimiter=',', fmt='%d')
           
            logger.warning(f"Screen tearing detected! Low row correlation frames saved to {sesName+'_low_row_corr_frames.csv'}")
        
        logger.info(f"Saving motion corrected tiffs and calculating row correlation finished in ({time.perf_counter() - t0:.2f} s)")

        #%% save the first tiff file in c-order memmap 
        # save the tiff file to C order memmap file
        t0 = time.perf_counter() # time for saving c-order memmap

        tiff_files = [
            os.path.join(output_tiff_dir, f)
            for f in os.listdir(output_tiff_dir)
            if f.lower().endswith((".tif", ".tiff"))
        ]
        #print(tiff_files)

        if not 'bord_px' in locals():
            bord_px = 0

        # save downsampled memmap only for now
        #fname_new = cm.save_memmap(tiff_files,base_name=sesName+'_', order='C',
        #                            border_to_0=bord_px)

        # spatial downsample
        fname_DS = cm.save_memmap(tiff_files,base_name=sesName+'_ds2_', order='C', resize_fact=(0.5,0.5,1),
                            border_to_0=bord_px)
        #print(tiff_files)

        
        # delete F_order memmap file and all individual c-order memmap file except for the first one
        pattern = re.compile(
            rf"^{re.escape(sesName)}.*_(\d{{4}})_d1"
        )

        for fname in os.listdir(out_path):
            match = pattern.match(fname)
            if match:
                idx = int(match.group(1))
                if idx != 0:  # keep sesName_0000_d1*
                    fullpath = os.path.join(out_path, fname)
                    print(f"Deleting {fullpath}")
                    os.remove(fullpath)

        logger.info(f"Saved C-order memmap in ({time.perf_counter() - t0:.2f} s)")

            
            #plt.figure()
            # plt.imshow(images[27362], cmap='gray'
            #            )
            # plt.show()
            # plt.savefig(os.path.join(out_path, sesName+'_frame_27362.png'))

        # clean up


        # ---- release F-order memmap cleanly ----
        # del images
        # del Yr
        # del fname_mc


        # gc.collect()

        # f_order_file = glob.glob(os.path.join(out_path, '*_order_F_*.mmap'))
        # if len(f_order_file)>0:
        #     os.remove(f_order_file[0])
        #     print('Deleted F order memmap file: {}'.format(f_order_file[0]))

# ---------------- REQUIRED ON WINDOWS ----------------
if __name__ == "__main__":
    mp.freeze_support()

    # process single folder
    root_path = r'Y:\HongliWang\Miniscope\ASDC001\ASDC001_260118'
    if_batch = False
    main(root_path, if_batch)

    # batch process
    #root_path = r'Y:\HongliWang\Miniscope\ASDC001'
    #if_batch = True
    #main(root_path, if_batch)


    ## spatially downsample the video 
    # orig_file = r'Y:\HongliWang\Miniscope\ASDC001\ASDC001_260113\temp\ASDC001_AB_ImgVideo_2026-01-13T09_07_50_rig__d1_600_d2_600_d3_1_order_C_frames_154469.mmap'

    # ds_factor = 2
    # Yr, dims, T = cm.load_memmap(orig_file)
    # d1, d2 = dims
    # images = Yr.T.reshape((T,) + dims, order='F')
    # os.environ['CAIMAN_DATA']=r'Y:\HongliWang\Miniscope\ASDC001\ASDC001_260113'
    # # ------------------------------------------------------------------
    # # Save new mmap
    # # ------------------------------------------------------------------
    # base, _ = os.path.splitext(orig_file)
    # ds_file = 'ASDC001_AB_ImgVideo_2026-01-13T09_07_50_rig_'
    # tiff_files=  glob.glob(r'Y:\HongliWang\Miniscope\ASDC001\ASDC001_260113\motion_corrected_tiffs\*.tif')

    
    # dsfile = cm.save_memmap(
    #     tiff_files,
    #     base_name=ds_file,
    #     resize_fact=(0.5,0.5,1),
    #     order='C'
    # )

    # Yr, dims, T = cm.load_memmap(dsfile)
    # d1, d2 = dims
    # images = Yr.T.reshape((T,) + dims, order='F')

    # print('Saved spatially downsampled mmap:')
    # print(ds_file)
    # print(f'New shape: {new_d1} x {new_d2}, frames={T}')