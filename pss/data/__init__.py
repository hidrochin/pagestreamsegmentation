"""
PSS data layer.

On-disk layout (all under ``data.root``, default ``datasets/pss/``):

    images/<doc_id>/page_000.png ...          rendered page images
    docs/<doc_id>.json                        single, type-labeled documents
    folders/<split>/<folder_id>.json          synthesized streams (page references)
    class_names.txt                           document types, one per line
    preprocessed_files_{train,val,test}.txt   index: "<folders/split/fid.json>\t<n_pages>"

Single-document JSON:
    {"doc_id": "...", "type": "invoice",
     "pages": [{"width": W, "height": H, "image": "images/<doc_id>/page_000.png",
                "words": [{"text": "Invoice", "box": [x0, y0, x1, y1]}, ...]}, ...]}
    (box in pixel coords; image path relative to data.root)

Folder (stream) JSON — references pages so 100k folders don't duplicate content:
    {"folder_id": "...",
     "pages": [{"doc_id": "...", "page_idx": 0, "boundary": 1, "type": "invoice"}, ...]}
"""

__all__ = ["PSSDataset", "collate_streams"]


def __getattr__(name):
    # Lazy so `python -m pss.data.synthesize` (pure stdlib) doesn't pull torch.
    if name == "PSSDataset":
        from pss.data.dataset import PSSDataset

        return PSSDataset
    if name == "collate_streams":
        from pss.data.collate import collate_streams

        return collate_streams
    raise AttributeError(name)
