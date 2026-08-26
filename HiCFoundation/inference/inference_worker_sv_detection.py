import os
import pickle
import shutil
import numpy as np
import torch
import torch.nn as nn
from ops.Logger import MetricLogger
from scipy.sparse import coo_matrix

from .external_aggregate import aggregate_raw_file, read_aggregated
from .disk_offdiag_writer import DiskOffDiagWriter

# Must match the hardcoded diagonal-scan cutoff in
# data_processing/inference_dataset_sv_detection.py ("if abs(i-j)>100: continue").
# args.bound is NOT a parameter of that dataset class.
DIAGONAL_SCAN_BOUND = 100

# Final threshold applied to the symmetrized prediction (original behaviour).
KEEP_THRESHOLD = 0.01

# How many raw off-diagonal records to buffer in RAM per chromosome before
# appending to its on-disk raw file. Small and constant regardless of total
# off-diagonal volume - the whole point of this design.
DISK_WRITE_BUFFER = 2_000_000

# How many raw records to sort+aggregate per in-memory chunk during external
# aggregation. This (not FLUSH_THRESHOLD, not total data size) is what bounds
# peak memory during finalization now. Lower = less peak RAM, more I/O passes.
EXTERNAL_AGG_CHUNK_RECORDS = 20_000_000


# ---------------------------------------------------------------------------
# per-window accumulation (band math unchanged; off-diagonal now goes to disk)
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
                      row_end, col_end, cur_output, off_diag_writer):
    """
    off_diag_writer: any object exposing .add(idx, vals) - a DiskOffDiagWriter
    in this version, writing raw (idx, val) records straight to disk instead
    of merging in RAM.
    """
    n_rows = row_end - row_start
    n_cols = col_end - col_start
    if n_rows <= 0 or n_cols <= 0:
        return
    band_width = mean_band.shape[1]
    col_size = off_diag_writer.col_size

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
        off_diag_writer.add(idx, np.ascontiguousarray(cur_output, dtype=np.float32).ravel())
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
        off_diag_writer.add(idx, np.asarray(cur_output, dtype=np.float32)[ri, ci])


# ---------------------------------------------------------------------------
# finalization (band math unchanged; off-diagonal now read from the
# externally-aggregated file instead of an in-RAM accumulator)
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


def _offdiag_upper(idx, mean, col_size, threshold):
    """
    Symmetrize the (already externally-aggregated, deduplicated) off-diagonal
    entries and return upper-triangle survivors. Same math as before; idx and
    mean now come from reading back the externally-aggregated file rather
    than an in-RAM OffDiagAccumulator, but by this point the data has already
    been deduplicated/shrunk by the external aggregation step, so it should
    comfortably fit in RAM for this final symmetrization pass even when the
    RAW pre-aggregation data did not.
    """
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


def _safe_filename(chrom):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(chrom))


def _finalize_and_spill(chrom, entry, band_half, partials_dir, raw_dir, agg_dir):
    mean_band = entry['mean']
    count_band = entry['count']
    row_size = entry['row_size']
    col_size = entry['col_size']
    raw_path = entry['raw_path']

    np.divide(mean_band, np.maximum(count_band, 1), out=mean_band)
    entry['count'] = None
    del count_band

    if row_size == col_size:
        _symmetrize_band_inplace(mean_band, band_half)
    band_rows, band_cols, band_vals = _extract_band_upper(
        mean_band, band_half, row_size, KEEP_THRESHOLD)
    entry['mean'] = None
    del mean_band

    # close the raw writer (flush any buffered-but-unwritten records), then
    # externally aggregate its on-disk file - memory bounded by
    # EXTERNAL_AGG_CHUNK_RECORDS, not by the raw file's total size.
    entry['off_diag_writer'].close()
    agg_path = aggregate_raw_file(raw_path, agg_dir, chunk_records=EXTERNAL_AGG_CHUNK_RECORDS,
                                   prefix=_safe_filename(chrom))
    os.remove(raw_path)

    idx_parts, mean_parts = [], []
    for chunk in read_aggregated(agg_path):
        idx_parts.append(chunk['idx'].astype(np.int64))
        mean_parts.append(chunk['mean'].astype(np.float32))
    od_idx = np.concatenate(idx_parts) if idx_parts else np.empty(0, dtype=np.int64)
    od_mean = np.concatenate(mean_parts) if mean_parts else np.empty(0, dtype=np.float32)
    os.remove(agg_path)

    od_rows, od_cols, od_vals = _offdiag_upper(od_idx, od_mean, col_size, KEEP_THRESHOLD)
    del od_idx, od_mean

    print("finish summarize %s prediction: band=%d off_diag=%d total=%d"
          % (chrom, band_rows.size, od_rows.size, band_rows.size + od_rows.size))

    rows = np.concatenate([band_rows, od_rows])
    cols = np.concatenate([band_cols, od_cols])
    vals = np.concatenate([band_vals, od_vals])
    del band_rows, band_cols, band_vals, od_rows, od_cols, od_vals

    prediction_sym = coo_matrix((vals, (rows, cols)), shape=(row_size, col_size))
    path = os.path.join(partials_dir, _safe_filename(chrom) + ".pkl")
    with open(path, "wb") as f:
        pickle.dump(prediction_sym, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def _new_chrom_entry(row_size, col_size, band_width, raw_dir, chrom):
    raw_path = os.path.join(raw_dir, _safe_filename(chrom) + "_raw.bin")
    return {
        "mean": np.zeros((row_size, band_width), dtype=np.float32),
        "count": np.zeros((row_size, band_width), dtype=np.uint16),
        "row_size": row_size,
        "col_size": col_size,
        "raw_path": raw_path,
        "off_diag_writer": DiskOffDiagWriter(raw_path, col_size, buffer_capacity=DISK_WRITE_BUFFER),
    }


def inference_worker(model, data_loader, log_dir=None, args=None):
    """
    model: model for inference
    data_loader: data loader for inference
    log_dir: log directory for inference
    args: arguments for inference

    Off-diagonal (far-off-diagonal SV signal) predictions are written raw to
    a per-chromosome file on disk as they're computed, instead of being
    merged in RAM - even the previous efficient in-RAM merge strategy could
    still exceed available RAM for a single very SV-dense chromosome. At
    finalize time, each chromosome's raw file is aggregated externally
    (chunked sort + k-way merge), bounding peak memory to a configurable
    chunk size regardless of how much raw data accumulated for that
    chromosome. Chromosomes are still allocated lazily and finalized/spilled
    to disk as soon as the window stream moves past them (relying on
    Inference_Dataset's guaranteed contiguous per-chromosome window
    ordering), so at most ~1-2 chromosomes' band/raw-writer state is open
    at any point in the main loop.
    """
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Inference: '
    print_freq = args.print_freq
    print("number of iterations: ", len(data_loader))
    dataset_shape_dict = data_loader.dataset.dataset_shape

    band_half = DIAGONAL_SCAN_BOUND + max(args.input_row_size, args.input_col_size)
    band_width = 2 * band_half + 1

    work_root = os.path.join(log_dir or ".", "_sv_work")
    partials_dir = os.path.join(work_root, "chrom_partials")
    raw_dir = os.path.join(work_root, "raw_offdiag")
    agg_dir = os.path.join(work_root, "agg_offdiag")
    for d in (partials_dir, raw_dir, agg_dir):
        os.makedirs(d, exist_ok=True)

    output_dict = {}
    finalized_paths = {}
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

            if prev_chr is not None and chr != prev_chr and prev_chr not in finalized_chroms:
                path = _finalize_and_spill(prev_chr, output_dict.pop(prev_chr), band_half,
                                            partials_dir, raw_dir, agg_dir)
                finalized_paths[prev_chr] = path
                finalized_chroms.add(prev_chr)
                print(f"Completed {prev_chr}")

            if chr in finalized_chroms:
                print(f"WARNING: chromosome {chr} reappeared after being finalized; "
                      f"this violates the expected contiguous window ordering. "
                      f"Re-opening it - its earlier spilled result will be replaced.")
                finalized_chroms.discard(chr)
                finalized_paths.pop(chr, None)

            if chr not in output_dict:
                row_size, col_size = dataset_shape_dict[chr]
                output_dict[chr] = _new_chrom_entry(row_size, col_size, band_width, raw_dir, chr)

            entry = output_dict[chr]
            _accumulate_patch(entry['mean'], entry['count'], band_half,
                              row_start, col_start, row_end, col_end,
                              cur_output, entry['off_diag_writer'])
            prev_chr = chr

    for chrom in list(output_dict.keys()):
        path = _finalize_and_spill(chrom, output_dict.pop(chrom), band_half,
                                    partials_dir, raw_dir, agg_dir)
        finalized_paths[chrom] = path
        print(f"Completed {chrom}")

    final_dict = {}
    grand_total = 0
    for chrom, path in finalized_paths.items():
        with open(path, "rb") as f:
            mat = pickle.load(f)
        final_dict[chrom] = mat
        grand_total += mat.nnz

    shutil.rmtree(work_root, ignore_errors=True)

    print("total nonzero pixels across all chromosomes: %d (~%.2f GB in the output pkl)"
          % (grand_total, grand_total * 12 / 1e9))
    return final_dict