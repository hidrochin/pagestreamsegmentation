"""
Prompt/completion construction for the decoder PSS track, plus parsing of the
model's JSON output back into plain boundary/type predictions.

Two ``decoder.context_mode`` formulations (see pss/decoder/config.py):
  - "pair": reproduces the source paper's formulation (arXiv:2408.11981,
    Listing 1) — previous + current page text -> a single boundary decision.
    Extended here to also request the document ``type`` (the paper only did
    binary PSS; this repo's task per CLAUDE.md is boundary *and* type).
  - "window": one prompt holds up to ``data.max_pages`` pages -> a JSON array
    of per-page {boundary, type} predictions in one shot. The paper explicitly
    left "give the decoder more than a page pair" as unexplored future work
    (Sec 5.1.1); this mirrors the repo's own I1 joint-context encoder variant
    and uses decoder context windows (8K+) that dwarf the paper's per-model
    limit that forced the pair formulation in the first place.

Text serialization preserves 2D layout via whitespace (per the paper's own
approach, citing [Wan+23]) — built directly from ``docs/*.json``'s
``page["words"]`` (text + pixel box), no OCR re-run needed.
"""

import json
import re

# -- layout-preserving text serialization ----------------------------------------


def _estimate_char_unit(words):
    """Median page-unit width per character, used to convert horizontal gaps
    between words into a whitespace run. Falls back to a generic default when
    there's too little text to estimate from."""
    widths = [
        (w["box"][2] - w["box"][0]) / max(1, len(w["text"]))
        for w in words
        if w.get("text")
    ]
    if not widths:
        return 6.0
    widths.sort()
    return max(widths[len(widths) // 2], 1.0)


def serialize_page_text(page, row_tol_frac=0.01):
    """Render a page's OCR words into a single string that preserves rough 2D
    layout via whitespace — words are grouped into rows by vertical center,
    sorted left-to-right within a row, and horizontal gaps become proportional
    runs of spaces. ``page``: {"width","height","words":[{"text","box"}]}."""
    words = page.get("words", [])
    if not words:
        return ""
    height = max(1, page.get("height", 1))
    row_tol = max(1.0, height * row_tol_frac)
    char_unit = _estimate_char_unit(words)

    items = sorted(
        (
            (0.5 * (w["box"][1] + w["box"][3]), w["box"][0], w["box"][2], w["text"])
            for w in words
        ),
        key=lambda t: (t[0], t[1]),
    )

    rows = []
    for ycenter, x0, x1, text in items:
        if rows and abs(ycenter - rows[-1][0]) <= row_tol:
            rows[-1][1].append((x0, x1, text))
        else:
            rows.append((ycenter, [(x0, x1, text)]))

    lines = []
    for _, row_words in rows:
        row_words.sort(key=lambda t: t[0])
        parts, prev_x1 = [], None
        for x0, x1, text in row_words:
            if prev_x1 is not None:
                gap_chars = max(1, round((x0 - prev_x1) / char_unit))
                parts.append(" " * gap_chars)
            parts.append(text)
            prev_x1 = x1
        lines.append("".join(parts))
    return "\n".join(lines)


# -- prompt templates --------------------------------------------------------------

_SYSTEM_HEADER = (
    "You are a skilled document reviewer. Given extracted text from pages of "
    "documents, your task is to determine whether each page starts a new "
    "document or continues from the previous one, and, when a page starts a "
    "new document, to classify that document's type.\n\n"
    "Document types: {types}\n"
)

_PAIR_TEMPLATE = (
    _SYSTEM_HEADER
    + "\nYou will be presented with the text of the current page and the text "
    "of the preceding page (prior text is empty if this is the first page of "
    "the stream).\n\n"
    "Prior text:\n###\n{prior}\n###\n"
    "Page text:\n###\n{page}\n###\n\n"
    'Output your prediction as a JSON object with keys "boundary" (1 if the '
    'page starts a new document, 0 if it continues the previous one) and '
    '"type" (one of the document types above if boundary=1, else null). Do '
    "not provide any explanation, additional information, or punctuation. "
    "Simply provide the JSON object."
)

_WINDOW_TEMPLATE = (
    _SYSTEM_HEADER
    + "\nYou will be presented with {n} consecutive pages from a stream, each "
    "labeled with its page number (0-indexed within this excerpt). Page 0's "
    "prior context, if any, is not shown; treat it as a continuation unless "
    "you have another reason to believe otherwise.\n\n"
    "{pages}\n"
    'Output your prediction as a JSON array with exactly {n} objects, one per '
    'page in order, each with keys "page" (its page number), "boundary" (1 if '
    'that page starts a new document, 0 if it continues the previous one), and '
    '"type" (one of the document types above if boundary=1, else null). Do not '
    "provide any explanation, additional information, or punctuation. Simply "
    "provide the JSON array."
)


def build_pair_prompt(prior_text, page_text, type_names):
    return _PAIR_TEMPLATE.format(
        types=", ".join(type_names), prior=prior_text, page=page_text
    )


def build_pair_completion(boundary, type_name):
    return json.dumps({"boundary": int(boundary), "type": type_name if boundary else None})


def build_window_prompt(page_texts, type_names):
    n = len(page_texts)
    pages = "\n".join(
        f"Page {i} text:\n###\n{t}\n###" for i, t in enumerate(page_texts)
    )
    return _WINDOW_TEMPLATE.format(types=", ".join(type_names), n=n, pages=pages)


def build_window_completion(boundaries, type_names_per_page):
    return json.dumps(
        [
            {"page": i, "boundary": int(b), "type": t if b else None}
            for i, (b, t) in enumerate(zip(boundaries, type_names_per_page))
        ]
    )


# -- output parsing (with the paper's malformed-JSON fallback) ---------------------

_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_ARR_RE = re.compile(r"\[.*\]", re.DOTALL)


def parse_pair_completion(text, type2id):
    """Returns (boundary: int, type_id: int|-100). Malformed/off-schema JSON
    defaults to boundary=0 (the paper's exact fallback: 'current page
    continues the previous document'), type_id=-100 (ignored by metrics)."""
    m = _OBJ_RE.search(text or "")
    if not m:
        return 0, -100
    try:
        obj = json.loads(m.group(0))
        boundary = int(obj.get("boundary", 0))
        boundary = 1 if boundary == 1 else 0
        type_id = type2id.get(obj.get("type"), -100) if boundary else -100
        return boundary, type_id
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0, -100


def parse_window_completion(text, n_pages, type2id):
    """Returns list[(boundary, type_id)] of length n_pages. Falls back to
    boundary=0/type=-100 per-page on malformed JSON, wrong length, or a
    missing page entry — the joint-mode analogue of the paper's fallback."""
    fallback = [(0, -100)] * n_pages
    m = _ARR_RE.search(text or "")
    if not m:
        return fallback
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return fallback
    if not isinstance(arr, list):
        return fallback

    by_page = {}
    for item in arr:
        if not isinstance(item, dict) or "page" not in item:
            continue
        try:
            idx = int(item["page"])
        except (TypeError, ValueError):
            continue
        boundary = 1 if int(item.get("boundary", 0)) == 1 else 0
        type_id = type2id.get(item.get("type"), -100) if boundary else -100
        by_page[idx] = (boundary, type_id)

    return [by_page.get(i, (0, -100)) for i in range(n_pages)]
