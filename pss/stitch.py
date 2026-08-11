"""Stitch overlapping sliding-window predictions back into one per-stream
sequence, averaging logits over pages where multiple windows overlap.

Shared by ``pss.evaluate`` (scores stitched predictions against ground truth) and
``pss.infer`` (production, no ground truth — only ``add_batch``/``stitch`` are used).
"""

from collections import defaultdict


class StreamAccumulator:
    def __init__(self):
        self.bd_sum = defaultdict(dict)
        self.ty_sum = defaultdict(dict)
        self.cnt = defaultdict(dict)
        self.bd_true = defaultdict(dict)
        self.ty_true = defaultdict(dict)

    def add_batch(self, meta, bl, tl, by=None, ty=None):
        """meta: list[{"folder_id", "start", "length"}], one per batch item —
        ``length`` must be the REAL (unpadded) page count of that window, so
        fixed-width inference windows padded with page_mask=0 (see pss.infer)
        are naturally excluded just by not being iterated over here.

        bl/tl: [B, P, ...] boundary/type logits (CPU tensors). by/ty: optional
        [B, P] ground-truth labels (pss.evaluate only; omit for production
        inference, where there's nothing to compare against).
        """
        for i, m in enumerate(meta):
            fid, start, length = m["folder_id"], m["start"], m["length"]
            for j in range(length):
                p = start + j
                self.bd_sum[fid][p] = self.bd_sum[fid].get(p, 0) + bl[i, j]
                self.ty_sum[fid][p] = self.ty_sum[fid].get(p, 0) + tl[i, j]
                self.cnt[fid][p] = self.cnt[fid].get(p, 0) + 1
                if by is not None:
                    self.bd_true[fid][p] = int(by[i, j])
                if ty is not None:
                    self.ty_true[fid][p] = int(ty[i, j])

    def stream_ids(self):
        return list(self.bd_sum.keys())

    def stitch(self, fid):
        """Return (idxs, pred_boundary, pred_type) for one stream: page indices
        sorted ascending, with per-page logits averaged over overlapping windows
        and argmax'd."""
        idxs = sorted(self.bd_sum[fid])
        pred_bd = [int((self.bd_sum[fid][p] / self.cnt[fid][p]).argmax()) for p in idxs]
        pred_ty = [int((self.ty_sum[fid][p] / self.cnt[fid][p]).argmax()) for p in idxs]
        return idxs, pred_bd, pred_ty
