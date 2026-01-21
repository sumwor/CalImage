import numpy as np
import caiman as cm

def convert_F_to_C_memmap(
    fname_f,
    fname_c=None,
    chunk_frames=200,
    dtype=np.float32
    ):
    """
    Convert a large CaImAn F-order memmap to C-order safely.

    Parameters
    ----------
    fname_f : str
        Input F-order .mmap file
    fname_c : str or None
        Output C-order .mmap file (auto-generated if None)
    chunk_frames : int
        Number of frames processed per chunk
    dtype : numpy dtype
        Data type (must match input)
    """

    Yr, dims, T = cm.load_memmap(fname_f)
    d1, d2 = dims

    if fname_c is None:
        fname_c = fname_f.replace("order_F", "order_C")

    print(f'Converting F → C memmap')
    print(f'Input:  {fname_f}')
    print(f'Output: {fname_c}')
    print(f'Shape:  ({T}, {d1}, {d2})')

    # Create output memmap
    Yc = np.memmap(
        fname_c,
        dtype=dtype,
        mode='w+',
        shape=(T, d1, d2),
        order='C'
    )

    for start in range(0, T, chunk_frames):
        end = min(start + chunk_frames, T)

        # Read F-order chunk
        chunk = (
            Yr[:, start:end]
            .reshape(d1, d2, end - start, order='C')
            .transpose(2, 0, 1)
        )

        Yc[start:end] = chunk
        Yc.flush()

        if start % (chunk_frames * 10) == 0:
            print(f'  written frames {start}–{end}')

    del Yc
    print('Conversion complete.')

    return fname_c

def frame_row_corr(frame, ref):
    """
    Compute average correlation between consecutive rows of a frame.
    Lower correlation → likely shifted rows.
    """
    nRows = frame.shape[0]
    corr_vals = []
    for r in range(1, nRows):
        corr = np.corrcoef(frame[r, :], ref[r, :])[0,1]
        corr_vals.append(corr)
    return np.mean(corr_vals)

    # idx = 9775
    # corr_vals_f1 = frame_row_corr(images[idx,:,:], ref)

    # idx = 1 # normal ref
    # corr_vals_ctrl = frame_row_corr(images[idx,:,:], ref)

    # idx = 27870
    # corr_vals_f2 = frame_row_corr(images[idx,:,:], ref)

    # print(corr_vals_f1)
    # print(corr_vals_f2)
    # print(corr_vals_ctrl)

