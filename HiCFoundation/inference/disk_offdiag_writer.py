import numpy as np
from .external_aggregate import RAW_DTYPE


class DiskOffDiagWriter:
    """
    Same .add(idx, vals) interface as OffDiagAccumulator, so _accumulate_patch
    doesn't need to change at all - only what object gets passed to it. But
    instead of merging in RAM (bounded by the eventual AGGREGATED size, which
    can still be too large for a single very SV-dense chromosome), this just
    buffers a modest amount and appends raw, unaggregated records straight to
    disk. Aggregation happens later, externally, with memory bounded by chunk
    size regardless of the raw file's total size.
    """

    def __init__(self, path, col_size, buffer_capacity=2_000_000):
        self.path = path
        self.col_size = int(col_size)
        self.buffer_capacity = buffer_capacity
        self._idx_buf = np.empty(buffer_capacity, dtype=np.int64)
        self._val_buf = np.empty(buffer_capacity, dtype=np.float32)
        self._n = 0
        self._file = open(path, "wb")
        self._closed = False

    def add(self, idx, vals):
        n = idx.size
        if n == 0:
            return
        if n > self.buffer_capacity:
            # rare (would need a single window's out-of-band pixel count to
            # exceed the buffer) - write directly, bypassing the buffer
            self.flush()
            self._write_records(idx, vals)
            return
        if self._n + n > self.buffer_capacity:
            self.flush()
        self._idx_buf[self._n:self._n + n] = idx
        self._val_buf[self._n:self._n + n] = vals
        self._n += n

    def _write_records(self, idx, vals):
        rec = np.empty(idx.size, dtype=RAW_DTYPE)
        rec['idx'] = idx
        rec['val'] = vals
        rec.tofile(self._file)

    def flush(self):
        if self._n == 0:
            return
        self._write_records(self._idx_buf[:self._n], self._val_buf[:self._n])
        self._n = 0

    def close(self):
        if self._closed:
            return
        self.flush()
        self._file.close()
        self._closed = True
