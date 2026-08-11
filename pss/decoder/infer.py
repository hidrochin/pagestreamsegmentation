"""
Production inference: decoder analogue of ``pss/infer.py`` — runs a trained
decoder checkpoint over a live, unlabeled page stream (same
``{"doc_id", "pages":[...]}`` JSON schema ``pss.infer`` reads) and prints/writes
per-page boundary + type predictions. No ground truth required (unlike
``pss.decoder.evaluate``).

Uses ``infer.window_pages``/``infer.window_stride`` (independent of
``data.max_pages``/``window_stride`` used at training time, mirroring the
encoder pipeline's deployment-hardware-constraint pattern — see
``pss/config.py``'s matching warning) and majority-vote stitches overlapping
windows the same way ``pss.decoder.evaluate`` does.

    python -m pss.decoder.infer --config=configs/pss_decoder.yaml \
        infer.checkpoint=pss_runs/decoder_exp/checkpoints/<run_id>/last \
        infer.stream=stream.json

Requires ``.venv-decoder`` on a CUDA GPU.
"""

import json

from pss.data.resolve import read_class_names
from pss.data.windowing import build_stream_windows
from pss.decoder.config import get_config
from pss.decoder.generate import generate_completions
from pss.decoder.prompt import (
    build_pair_prompt,
    build_window_prompt,
    parse_pair_completion,
    parse_window_completion,
    serialize_page_text,
)
from pss.decoder.stitch import VoteAccumulator


def run_stream(cfg, model, tokenizer, stream_doc, type_names):
    """stream_doc: {"doc_id", "pages": [...]} in the docs/*.json page schema
    (``type`` is ignored — that's what we're predicting). Returns a list of
    {"page_idx", "pred_boundary", "pred_type"} in stream order."""
    type2id = {c: i for i, c in enumerate(type_names)}
    id2type = {i: c for c, i in type2id.items()}
    pages = stream_doc["pages"]
    fid = stream_doc.get("doc_id", "stream")
    acc = VoteAccumulator()

    windows = (
        [(i, 1) for i in range(len(pages))]
        if cfg.decoder.context_mode == "pair"
        else build_stream_windows(len(pages), cfg.infer.window_pages, cfg.infer.window_stride)
    )

    prompts = []
    for start, length in windows:
        if cfg.decoder.context_mode == "pair":
            prior_text = serialize_page_text(pages[start - 1]) if start > 0 else ""
            page_text = serialize_page_text(pages[start])
            prompts.append(build_pair_prompt(prior_text, page_text, type_names))
        else:
            page_texts = [serialize_page_text(pages[start + j]) for j in range(length)]
            prompts.append(build_window_prompt(page_texts, type_names))

    chat_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    completions = generate_completions(cfg, model, tokenizer, chat_prompts)

    for (start, length), raw in zip(windows, completions):
        if cfg.decoder.context_mode == "pair":
            b, t = parse_pair_completion(raw, type2id)
            preds = [(b, t)]
        else:
            preds = parse_window_completion(raw, length, type2id)
        acc.add_window(fid, start, [p[0] for p in preds], [p[1] for p in preds])

    idxs, pred_bd, pred_ty = acc.stitch(fid)
    return [
        {"page_idx": p, "pred_boundary": b, "pred_type": id2type.get(t)}
        for p, b, t in zip(idxs, pred_bd, pred_ty)
    ]


def main():
    cfg = get_config()
    assert cfg.infer.checkpoint, "pass infer.checkpoint=<trained LoRA adapter dir>"
    assert cfg.infer.stream, "pass infer.stream=<path to a stream json>"

    if cfg.decoder.modality == "vision":
        raise NotImplementedError(
            "vision-modality inference needs image-aware generation — see "
            "pss/decoder/train.py's matching NotImplementedError note."
        )

    from pss.decoder.evaluate import load_eval_model

    model, tokenizer = load_eval_model(cfg)
    type_names = read_class_names(cfg.data.root)

    with open(cfg.infer.stream, encoding="utf-8") as fp:
        stream_doc = json.load(fp)

    preds = run_stream(cfg, model, tokenizer, stream_doc, type_names)

    if cfg.infer.out:
        with open(cfg.infer.out, "w", encoding="utf-8") as fp:
            json.dump(preds, fp, indent=2)
        print(f"[decoder.infer] wrote {len(preds)} page predictions to {cfg.infer.out}")
    else:
        print(json.dumps(preds, indent=2))


if __name__ == "__main__":
    main()
