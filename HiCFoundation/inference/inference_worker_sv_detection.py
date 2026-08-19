import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from ops.Logger import MetricLogger
from scipy.sparse import coo_matrix

# Must match the hardcoded diagonal-scan cutoff in
# data_processing/inference_dataset_sv_detection.py ("if abs(i-j)>100: continue").
# args.bound is NOT a parameter of that dataset class.
DIAGONAL_SCAN_BOUND = 100

# Raw out-of-band (row, col, value) entries buffered per chromosome before
# compression. Lower = less peak RAM, slightly more CPU.
FLUSH_THRESHOLD = 4_000_000

# Final threshold applied to the symmetrized prediction (original behaviour).
KEEP_THRESHOLD = 0.01

# Optional: drop out-of-band pixels whose averaged value is <= this BEFORE
# symmetrizing. 0.0 = exact (default).
MIN_OFFDIAG_VALUE = 0.0


# ---------------------------------------------------------------------------
# grouping helper
# ---------------------------------------------------------------------------
def _group_sum(keys, value_arrays):
    if keys.size == 0:
        return keys, list(value_arrays)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    value_arrays = [v[order] for v in value_arrays]
    del order
    is_start = np.empty(keys.size, dtype=bool)
    is_start[0] = True
    np.not_equal(keys[1:], keys[:-1], out=is_start[1:])
    starts = np.flatnonzero(is_start)
    del is_start
    out_keys = keys[starts]
    out_vals = [np.add.reduceat(v, starts) for v in value_arrays]
    return out_keys, out_vals


# ---------------------------------------------------------------------------
# off-diagonal accumulator
# ---------------------------------------------------------------------------
class OffDiagAccumulator:
    MIN_CAPACITY = 1 << 20

    def __init__(self, col_size):
        self.col_size = int(col_size)
        self._buf_idx = None
        self._buf_val = None
        self._cap = 0
        self._n = 0
        self._runs = []

    def _alloc(self, cap):
        cap = int(max(cap, self.MIN_CAPACITY))
        self._buf_idx = np.empty(cap, dtype=np.int64)
        self._buf_val = np.empty(cap, dtype=np.float32)
        self._cap = cap
        self._n = 0

    def add(self, idx, vals):
        n = idx.size
        if n == 0:
            return
        if self._buf_idx is None:
            self._alloc(FLUSH_THRESHOLD)
        if self._n + n > self._cap:
            self.flush()
            self._alloc(FLUSH_THRESHOLD)
        self._buf_idx[self._n:self._n + n] = idx
        self._buf_val[self._n:self._n + n] = vals
        self._n += n

    def flush(self):
        if self._n == 0:
            return
        idx = self._buf_idx[:self._n].copy()
        vals = self._buf_val[:self._n].copy()
        self._buf_idx = None
        self._buf_val = None
        self._cap = 0
        self._n = 0
        cnt = np.ones(vals.size, dtype=np.float32)
        keys, (s, c) = _group_sum(idx, [vals, cnt])
        self._push_run((keys, s, c))

    def _push_run(self, run):
        self._runs.append(run)
        while len(self._runs) >= 2 and self._runs[-1][0].size >= self._runs[-2][0].size:
            b = self._runs.pop()
            a = self._runs.pop()
            self._runs.append(self._merge(a, b))

    @staticmethod
    def _merge(a, b):
        keys, (s, c) = _group_sum(np.concatenate([a[0], b[0]]),
                                  [np.concatenate([a[1], b[1]]),
                                   np.concatenate([a[2], b[2]])])
        return (keys, s, c)

    def finalize(self):
        self.flush()
        while len(self._runs) > 1:
            b = self._runs.pop()
            a = self._runs.pop()
            self._runs.append(self._merge(a, b))
        if not self._runs:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        idx, vsum, vcnt = self._runs.pop()
        mean = (vsum / np.maximum(vcnt, 1)).astype(np.float32, copy=False)
        del vsum, vcnt
        keep = mean > MIN_OFFDIAG_VALUE
        return idx[keep], mean[keep]


# ---------------------------------------------------------------------------
# per-window accumulation
# ---------------------------------------------------------------------------
def _band_add(mean_band, count_band, row_start, col_start, n_rows, n_cols,
              band_half, cur_output):
    W = mean_band.shape[1]
    if n_cols <= W - 1:
        base = row_start * (W - 1) + col_start + band_half
        flat_m = mean_band.reshape(-1)
        flat_c = count_band.reshape(-1)
        shape = (n_rows, n_cols)
        np.lib.stride_tricks.as_strided(
            flat_m[base:], shape=shape,
            strides=((W - 1) * flat_m.itemsize, flat_m.itemsize))[...] += cur_output
        np.lib.stride_tricks.as_strided(
            flat_c[base:], shape=shape,
            strides=((W - 1) * flat_c.itemsize, flat_c.itemsize))[...] += 1
        return
    for k in range(n_rows):
        r = row_start + k
        b0 = col_start - r + band_half
        mean_band[r, b0:b0 + n_cols] += cur_output[k]
        count_band[r, b0:b0 + n_cols] += 1


def _accumulate_patch(mean_band, count_band, band_half, row_start, col_start,
                      row_end, col_end, cur_output, off_diag):
    n_rows = row_end - row_start
    n_cols = col_end - col_start
    if n_rows <= 0 or n_cols <= 0:
        return
    band_width = mean_band.shape[1]
    col_size = off_diag.col_size

    d_min = col_start - (row_end - 1)
    d_max = (col_end - 1) - row_start

    if d_min + band_half >= 0 and d_max + band_half < band_width:
        _band_add(mean_band, count_band, row_start, col_start,
                  n_rows, n_cols, band_half, cur_output)
        return

    if d_max + band_half < 0 or d_min + band_half >= band_width:
        rows = np.arange(row_start, row_end, dtype=np.int64)
        cols = np.arange(col_start, col_end, dtype=np.int64)
        idx = (rows[:, None] * col_size + cols[None, :]).ravel()
        off_diag.add(idx, np.ascontiguousarray(cur_output, dtype=np.float32).ravel())
        return

    for k in range(n_rows):
        r = row_start + k
        b0 = col_start - r + band_half
        lo = max(b0, 0)
        hi = min(b0 + n_cols, band_width)
        if hi > lo:
            mean_band[r, lo:hi] += cur_output[k, lo - b0:hi - b0]
            count_band[r, lo:hi] += 1

    kk = np.arange(n_rows, dtype=np.int64)[:, None]
    cc = np.arange(n_cols, dtype=np.int64)[None, :]
    band_pos = (col_start - row_start + band_half) + cc - kk
    outside = (band_pos < 0) | (band_pos >= band_width)
    if outside.any():
        ri, ci = np.nonzero(outside)
        idx = (row_start + ri) * np.int64(col_size) + (col_start + ci)
        off_diag.add(idx, np.asarray(cur_output, dtype=np.float32)[ri, ci])


# ---------------------------------------------------------------------------
# finalization
# ---------------------------------------------------------------------------
def _symmetrize_band_inplace(mean_band, band_half):
    R, W = mean_band.shape
    for k in range(band_half):
        kp = W - 1 - k
        dp = band_half - k
        if dp >= R:
            continue
        s = mean_band[dp:R, k] + mean_band[0:R - dp, kp]
        s *= 0.5
        mean_band[0:R - dp, kp] = s


def _extract_band_upper(mean_band, band_half, row_size, threshold):
    W = mean_band.shape[1]
    rows_l, cols_l, vals_l = [], [], []
    for k in range(band_half, W):
        d = k - band_half
        hi = row_size - d
        if hi <= 0:
            break
        col = mean_band[0:hi, k]
        nz = np.flatnonzero(col > threshold)
        if nz.size:
            rows_l.append(nz.astype(np.int32))
            cols_l.append((nz + d).astype(np.int32))
            vals_l.append(col[nz].astype(np.float32, copy=False))
    if not rows_l:
        return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32))
    return np.concatenate(rows_l), np.concatenate(cols_l), np.concatenate(vals_l)


def _offdiag_upper(idx, mean, col_size, threshold):
    if idx.size == 0:
        return (np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32))

    C = np.int64(col_size)
    r = idx // C
    idx_t = idx - r * C
    idx_t *= C
    idx_t += r
    del r
    key = np.minimum(idx, idx_t)
    del idx_t

    key, (total,) = _group_sum(key, [mean])
    total *= 0.5

    keep = total > threshold
    key = key[keep]
    total = total[keep]
    del keep

    lo = (key // C).astype(np.int32)
    hi = (key - lo.astype(np.int64) * C).astype(np.int32)
    return lo, hi, total.astype(np.float32, copy=False)


def _finalize_chrom(entry, band_half, chrom):
    mean_band = entry['mean']
    count_band = entry['count']
    row_size = entry['row_size']
    col_size = entry['col_size']

    np.divide(mean_band, np.maximum(count_band, 1), out=mean_band)
    entry['count'] = None
    del count_band

    if row_size == col_size:
        _symmetrize_band_inplace(mean_band, band_half)
    band_rows, band_cols, band_vals = _extract_band_upper(
        mean_band, band_half, row_size, KEEP_THRESHOLD)
    entry['mean'] = None
    del mean_band

    od_idx, od_mean = entry['off_diag'].finalize()
    entry['off_diag'] = None
    od_rows, od_cols, od_vals = _offdiag_upper(
        od_idx, od_mean, col_size, KEEP_THRESHOLD)
    del od_idx, od_mean

    print("finish summarize %s prediction: band=%d off_diag=%d total=%d"
          % (chrom, band_rows.size, od_rows.size, band_rows.size + od_rows.size))

    rows = np.concatenate([band_rows, od_rows])
    cols = np.concatenate([band_cols, od_cols])
    vals = np.concatenate([band_vals, od_vals])
    del band_rows, band_cols, band_vals, od_rows, od_cols, od_vals

    return coo_matrix((vals, (rows, cols)), shape=(row_size, col_size))


# ---------------------------------------------------------------------------
# streaming per-chromosome finalize+write-to-disk
# ---------------------------------------------------------------------------
def _safe_filename(chrom):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(chrom))


def _finalize_and_spill(chrom, entry, band_half, partials_dir):
    """
    Finalize one chromosome's accumulators (same math as before) and
    immediately pickle the small, already-compressed result to disk, then
    let the caller drop the (much larger) raw entry from RAM.
    """
    prediction_sym = _finalize_chrom(entry, band_half, chrom)
    path = os.path.join(partials_dir, _safe_filename(chrom) + ".pkl")
    with open(path, "wb") as f:
        pickle.dump(prediction_sym, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _new_chrom_entry(row_size, col_size, band_width):
    return {
        "mean": np.zeros((row_size, band_width), dtype=np.float32),
        "count": np.zeros((row_size, band_width), dtype=np.uint16),
        "row_size": row_size,
        "col_size": col_size,
        "off_diag": OffDiagAccumulator(col_size),
    }


def inference_worker(model, data_loader, log_dir=None, args=None):
    """
    model: model for inference
    data_loader: data loader for inference
    log_dir: log directory for inference
    args: arguments for inference

    Chromosomes are allocated lazily and finalized+spilled to disk as soon as
    the window stream moves past them, instead of allocating all chromosomes
    up front and only finalizing after the whole (multi-hour) loop finishes.
    This relies on Inference_Dataset building input_index chromosome-by-
    chromosome and the DataLoader using shuffle=False, so windows for a given
    chromosome are guaranteed contiguous - peak memory is therefore bounded
    to ~1-2 chromosomes' worth of accumulators (the active one, plus
    occasionally the next one starting mid-batch) instead of the whole
    genome's, regardless of how many chromosomes or how large the
    off-diagonal breakpoint set is.
    """
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Inference: '
    print_freq = args.print_freq
    print("number of iterations: ", len(data_loader))
    dataset_shape_dict = data_loader.dataset.dataset_shape

    band_half = DIAGONAL_SCAN_BOUND + max(args.input_row_size, args.input_col_size)
    band_width = 2 * band_half + 1

    partials_dir = os.path.join(log_dir or ".", "_chrom_partials")
    os.makedirs(partials_dir, exist_ok=True)

    output_dict = {}          # only currently-open chromosomes live here
    finalized_paths = {}      # chrom -> path to its spilled pkl
    finalized_chroms = set()
    prev_chr = None

    cutoff = 1000
    cutoff = torch.tensor(cutoff).float().cuda()
    log_cutoff = torch.log10(cutoff + 1).cuda()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        input, total_count, indexes = data
        input = input.cuda()
        total_count = total_count.cuda()
        total_count = total_count.float()
        model_dtype = next(model.parameters()).dtype
        input = input.to(model_dtype)
        with torch.no_grad(), torch.cuda.amp.autocast():
            output = model(input, total_count)
            output = output.float()

        output = output * log_cutoff
        output = torch.pow(10, output) - 1
        output = torch.clamp(output, min=0)

        output = output.detach().cpu().numpy()
        input = input.detach().cpu().numpy()
        chrs, row_starts, col_starts = indexes
        for i in range(len(output)):
            chr = chrs[i]
            row_start = max(0, int(row_starts[i]))
            col_start = max(0, int(col_starts[i]))
            current_shape = dataset_shape_dict[chr]
            row_end = min(row_start + args.input_row_size, current_shape[0])
            col_end = min(col_start + args.input_col_size, current_shape[1])
            current_input = input[i]

            if np.isnan(np.sum(current_input)):
                print("empty matrix:", chr, row_start, col_start)
                continue
            cur_output = output[i][:row_end - row_start, :col_end - col_start]

            # detect a transition away from the previously-active chromosome:
            # its windows are done (contiguous ordering guarantee), so
            # finalize+spill it now instead of waiting for the whole loop.
            if prev_chr is not None and chr != prev_chr and prev_chr not in finalized_chroms:
                path = _finalize_and_spill(prev_chr, output_dict.pop(prev_chr), band_half, partials_dir)
                finalized_paths[prev_chr] = path
                finalized_chroms.add(prev_chr)

            if chr in finalized_chroms:
                # should not happen given Inference_Dataset's ordering, but
                # guard against it rather than silently losing data
                print(f"WARNING: chromosome {chr} reappeared after being finalized; "
                      f"this violates the expected contiguous window ordering. "
                      f"Re-opening it - its earlier spilled result will be replaced.")
                finalized_chroms.discard(chr)
                finalized_paths.pop(chr, None)

            if chr not in output_dict:
                row_size, col_size = dataset_shape_dict[chr]
                output_dict[chr] = _new_chrom_entry(row_size, col_size, band_width)

            entry = output_dict[chr]
            _accumulate_patch(entry['mean'], entry['count'], band_half,
                              row_start, col_start, row_end, col_end,
                              cur_output, entry['off_diag'])
            prev_chr = chr
            print('Completed', chr)

    # finalize whatever is still open (normally just the last chromosome)
    for chrom in list(output_dict.keys()):
        path = _finalize_and_spill(chrom, output_dict.pop(chrom), band_half, partials_dir)
        finalized_paths[chrom] = path

    # reassemble the final dict from the (small, already-compressed) spilled
    # files, preserving the existing return contract for main_worker.py
    final_dict = {}
    grand_total = 0
    for chrom, path in finalized_paths.items():
        with open(path, "rb") as f:
            mat = pickle.load(f)
        final_dict[chrom] = mat
        grand_total += mat.nnz

    print("total nonzero pixels across all chromosomes: %d (~%.2f GB in the output pkl)"
          % (grand_total, grand_total * 12 / 1e9))
    return final_dict
