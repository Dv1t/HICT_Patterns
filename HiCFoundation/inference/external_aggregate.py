import heapq
import os
import numpy as np

# raw record format written by the disk-based off-diagonal writer:
# int64 idx, float32 val (12 bytes each)
RAW_DTYPE = np.dtype([('idx', '<i8'), ('val', '<f4')])
# run-file record format (post chunk-aggregation): idx, sum, count
RUN_DTYPE = np.dtype([('idx', '<i8'), ('sum', '<f4'), ('cnt', '<f4')])
# final output format: idx, mean
OUT_DTYPE = np.dtype([('idx', '<i8'), ('mean', '<f4')])


def _group_sum(idx, value_arrays):
    """Vectorized (numpy) sort+group-sum. value_arrays: list of arrays aligned with idx."""
    order = np.argsort(idx, kind="stable")
    idx = idx[order]
    value_arrays = [v[order] for v in value_arrays]
    is_start = np.empty(idx.size, dtype=bool)
    is_start[0] = True
    np.not_equal(idx[1:], idx[:-1], out=is_start[1:])
    starts = np.flatnonzero(is_start)
    out_idx = idx[starts]
    out_vals = [np.add.reduceat(v, starts) for v in value_arrays]
    return out_idx, out_vals


def _build_sorted_runs(raw_path, run_dir, chunk_records, prefix):
    """
    Phase 1: stream the raw file in chunks, sort+aggregate each chunk purely
    with numpy (fast, C-level), write each as a small sorted binary run file.
    Peak memory O(chunk_records), independent of the raw file's total size.
    """
    os.makedirs(run_dir, exist_ok=True)
    run_paths = []
    if not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
        return run_paths

    itemsize = RAW_DTYPE.itemsize
    with open(raw_path, "rb") as f:
        run_i = 0
        while True:
            buf = f.read(chunk_records * itemsize)
            if not buf:
                break
            n = len(buf) // itemsize
            arr = np.frombuffer(buf, dtype=RAW_DTYPE, count=n)
            idx = arr['idx'].astype(np.int64)
            val = arr['val'].astype(np.float32)
            g_idx, (g_sum, g_cnt) = _group_sum(idx, [val, np.ones_like(val)])
            out = np.empty(g_idx.size, dtype=RUN_DTYPE)
            out['idx'] = g_idx
            out['sum'] = g_sum
            out['cnt'] = g_cnt
            run_path = os.path.join(run_dir, f"{prefix}_run{run_i}.bin")
            out.tofile(run_path)
            run_paths.append(run_path)
            run_i += 1
    return run_paths


def _merge_two_run_files(path_a, path_b, out_path):
    """One vectorized merge of two sorted run files - no per-record python loop."""
    a = np.fromfile(path_a, dtype=RUN_DTYPE)
    b = np.fromfile(path_b, dtype=RUN_DTYPE)
    idx = np.concatenate([a['idx'], b['idx']])
    val = np.concatenate([a['sum'], b['sum']])
    cnt = np.concatenate([a['cnt'], b['cnt']])
    del a, b
    g_idx, (g_sum, g_cnt) = _group_sum(idx, [val, cnt])
    out = np.empty(g_idx.size, dtype=RUN_DTYPE)
    out['idx'] = g_idx
    out['sum'] = g_sum
    out['cnt'] = g_cnt
    out.tofile(out_path)
    return out_path


def _merge_all_runs(run_paths, work_dir, prefix):
    """
    Huffman-style merge: repeatedly combine the two SMALLEST run files (by
    file size) via a min-heap, which minimizes total bytes moved across all
    merges (small files get merged - and therefore re-read/re-written -
    fewer times than large ones). Each individual merge is one vectorized
    numpy operation, not a python per-record loop, so this stays fast
    regardless of how many total records there are.
    """
    if not run_paths:
        return None
    if len(run_paths) == 1:
        return run_paths[0]

    heap = [(os.path.getsize(p), i, p) for i, p in enumerate(run_paths)]
    heapq.heapify(heap)
    counter = len(run_paths)

    while len(heap) > 1:
        size_a, _, path_a = heapq.heappop(heap)
        size_b, _, path_b = heapq.heappop(heap)
        merged_path = os.path.join(work_dir, f"{prefix}_merge{counter}.bin")
        _merge_two_run_files(path_a, path_b, merged_path)
        os.remove(path_a)
        os.remove(path_b)
        heapq.heappush(heap, (os.path.getsize(merged_path), counter, merged_path))
        counter += 1

    return heap[0][2]


def aggregate_raw_file(raw_path, work_dir, chunk_records=20_000_000, prefix="chrom",
                        min_mean_threshold=0.0):
    """
    Raw (idx, val) binary file -> sorted, aggregated (idx, mean) binary file.
    Pure numpy throughout (no text formatting, no per-record python loops).
    Peak memory is bounded by chunk_records (phase 1) and by the size of the
    two largest runs being merged together (phase 2, which by construction
    only happens for the LAST merge - everything smaller merges earlier and
    cheaper), not by total raw data size.
    """
    run_dir = os.path.join(work_dir, "runs")
    run_paths = _build_sorted_runs(raw_path, run_dir, chunk_records, prefix)
    out_path = os.path.join(work_dir, f"{prefix}_aggregated.bin")

    final_run = _merge_all_runs(run_paths, run_dir, prefix)
    if final_run is None:
        open(out_path, "wb").close()
        return out_path

    arr = np.fromfile(final_run, dtype=RUN_DTYPE)
    os.remove(final_run)
    if os.path.isdir(run_dir):
        try:
            os.rmdir(run_dir)
        except OSError:
            pass

    mean = arr['sum'] / np.maximum(arr['cnt'], 1)
    keep = mean > min_mean_threshold
    out = np.empty(int(np.count_nonzero(keep)), dtype=OUT_DTYPE)
    out['idx'] = arr['idx'][keep]
    out['mean'] = mean[keep].astype(np.float32)
    out.tofile(out_path)
    return out_path


def read_aggregated(path, read_records=1_000_000):
    """Stream a finalized (idx, mean) aggregated file, yielding numpy array chunks."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, "rb") as f:
        itemsize = OUT_DTYPE.itemsize
        while True:
            buf = f.read(read_records * itemsize)
            if not buf:
                break
            n = len(buf) // itemsize
            yield np.frombuffer(buf, dtype=OUT_DTYPE, count=n)
