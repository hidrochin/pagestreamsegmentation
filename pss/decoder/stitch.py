"""Majority-vote stitching of per-page decoder predictions across overlapping
windows — the discrete-label analogue of ``pss/stitch.py``'s logit averaging,
which doesn't apply here since a generated JSON label isn't a probability
distribution that can be averaged across windows the way encoder logits are.
Shared by ``pss.decoder.evaluate`` (scores against ground truth) and
``pss.decoder.infer`` (production, no ground truth)."""

from collections import Counter, defaultdict


class VoteAccumulator:
    def __init__(self):
        self.bd_votes = defaultdict(lambda: defaultdict(list))
        self.ty_votes = defaultdict(lambda: defaultdict(list))
        self.bd_true = defaultdict(dict)
        self.ty_true = defaultdict(dict)

    def add_window(self, fid, start, boundary_preds, type_preds, true_boundary=None, true_type=None):
        """boundary_preds/type_preds: per-page predictions for one window,
        aligned (index j -> absolute page start+j). type_preds uses -100 for
        "no type predicted" (paper's convention: type only accompanies a
        predicted boundary) and is excluded from the vote."""
        for j, (b, t) in enumerate(zip(boundary_preds, type_preds)):
            p = start + j
            self.bd_votes[fid][p].append(b)
            if t != -100:
                self.ty_votes[fid][p].append(t)
            if true_boundary is not None:
                self.bd_true[fid][p] = true_boundary[j]
            if true_type is not None:
                self.ty_true[fid][p] = true_type[j]

    def stream_ids(self):
        return list(self.bd_votes.keys())

    def stitch(self, fid):
        """Return (idxs, pred_boundary, pred_type) for one stream: page
        indices sorted ascending, each majority-voted over every window that
        covered it. pred_type is -100 where no window ever predicted a type
        for that page (e.g. every covering window predicted boundary=0)."""
        idxs = sorted(self.bd_votes[fid])
        pred_bd = [Counter(self.bd_votes[fid][p]).most_common(1)[0][0] for p in idxs]
        pred_ty = [
            Counter(self.ty_votes[fid][p]).most_common(1)[0][0]
            if self.ty_votes[fid][p]
            else -100
            for p in idxs
        ]
        return idxs, pred_bd, pred_ty
