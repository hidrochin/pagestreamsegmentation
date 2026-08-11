"""Sliding-window index builder over a page stream — shared by ``PSSDataset``
(training/eval, windows read from a synthesized folder) and ``pss.infer``
(production, windows cut from a live stream) so both stay in lockstep.
"""


def build_stream_windows(n, max_pages, stride):
    """Return list of (start, length) windows covering ``n`` pages.

    A stream no longer than ``max_pages`` is a single window. Otherwise windows
    start at 0, stride, 2*stride, ...; the final window is right-aligned to the
    stream's end (so it can be shorter than ``max_pages`` but never overruns it,
    and no tail page is ever dropped).
    """
    if n <= max_pages:
        return [(0, n)]
    windows = []
    start = 0
    while start < n:
        length = min(max_pages, n - start)
        windows.append((start, length))
        if start + max_pages >= n:
            break
        start += stride
    return windows
