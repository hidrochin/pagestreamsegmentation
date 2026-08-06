"""
Render PDF pages to images (PyMuPDF).

Used in preprocessing (building the single-document corpus and, at inference,
turning an incoming stream PDF into per-page images). Not used during training.
"""

import os


def render_pdf(pdf_path, out_dir, dpi=150):
    """Rasterize every page of ``pdf_path`` to PNG under ``out_dir``.

    Returns list of (image_path, width, height), page order preserved.
    """
    import fitz  # PyMuPDF

    os.makedirs(out_dir, exist_ok=True)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    doc = fitz.open(pdf_path)
    out = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        path = os.path.join(out_dir, f"page_{i:03d}.png")
        pix.save(path)
        out.append((path, pix.width, pix.height))
    doc.close()
    return out
