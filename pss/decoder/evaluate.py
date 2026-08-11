"""
Evaluates a trained decoder checkpoint with the same page/document/stream
metrics as the encoder pipeline (``pss/metrics.py``, imported unmodified) so
the two tracks report directly comparable numbers.

    python -m pss.decoder.evaluate --config=configs/pss_decoder.yaml \
        infer.checkpoint=pss_runs/decoder_exp/checkpoints/<run_id>/last eval_mode=test

Run this against the paper's TABME++ sanity-check split *before* trusting
numbers on your own corpus — see pss/decoder/README.md's verification section.
Requires ``.venv-decoder`` on a CUDA GPU.
"""

from collections import Counter

from pss import metrics
from pss.data.resolve import read_class_names
from pss.decoder.config import get_config
from pss.decoder.dataset import build_examples
from pss.decoder.generate import build_prompts, generate_completions
from pss.decoder.prompt import parse_pair_completion, parse_window_completion
from pss.decoder.stitch import VoteAccumulator


def collect_predictions(cfg, model, tokenizer, mode):
    """Generates over every window in `mode`'s split, parses each completion,
    and majority-vote-stitches per-stream — the decoder analogue of
    pss.evaluate.collect_predictions. Returns the same dict shape (streams,
    type_pairs, all_pred, all_true) so callers can reuse pss.metrics as-is."""
    type_names = read_class_names(cfg.data.root)
    n_types = len(type_names)
    type2id = {c: i for i, c in enumerate(type_names)}

    examples = list(build_examples(cfg, mode))
    prompts = build_prompts(tokenizer, examples)
    completions = generate_completions(cfg, model, tokenizer, prompts)

    acc = VoteAccumulator()
    for ex, raw in zip(examples, completions):
        meta = ex["meta"]
        if cfg.decoder.context_mode == "pair":
            b, t = parse_pair_completion(raw, type2id)
            preds = [(b, t)]
        else:
            preds = parse_window_completion(raw, meta["length"], type2id)
        acc.add_window(
            meta["folder_id"],
            meta["start"],
            [p[0] for p in preds],
            [p[1] for p in preds],
            true_boundary=meta["true_boundary"],
            true_type=meta["true_type"],
        )

    streams, type_pairs, all_pred, all_true = [], [], [], []
    for fid in acc.stream_ids():
        idxs, pred_bd, pred_ty = acc.stitch(fid)
        true_bd = [acc.bd_true[fid][p] for p in idxs]
        streams.append((pred_bd, true_bd))
        all_pred.extend(pred_bd)
        all_true.extend(true_bd)

        # segment by TRUE boundaries (matches pss.evaluate.collect_predictions);
        # majority-vote the type within each true document from whichever pages
        # got a non-null type prediction. If the model never predicted a type
        # anywhere in this segment (e.g. it missed every boundary in it), fall
        # back to a value guaranteed != the true type so it scores as a miss
        # rather than crashing type_macro_f1 on the -100 sentinel.
        docs, cur = [], []
        for k in range(len(idxs)):
            if true_bd[k] == 1 and cur:
                docs.append(cur)
                cur = []
            cur.append(k)
        if cur:
            docs.append(cur)
        for d in docs:
            tv = acc.ty_true[fid][idxs[d[0]]]
            votes = [pred_ty[k] for k in d if pred_ty[k] != -100]
            pv = Counter(votes).most_common(1)[0][0] if votes else (tv + 1) % n_types
            type_pairs.append((tv, pv))

    return {
        "streams": streams,
        "type_pairs": type_pairs,
        "all_pred": all_pred,
        "all_true": all_true,
    }


def load_eval_model(cfg):
    """Load base model + attach a trained LoRA adapter for eval (mirrors
    pss.evaluate.load_eval_net). cfg.infer.checkpoint may also be a merged
    (post-export) HF dir — PeftModel.from_pretrained is a no-op-safe skip in
    that case only if the dir has no adapter_config.json; prefer pointing
    infer.checkpoint at the adapter dir for eval, export.out for serving."""
    from peft import PeftModel

    from pss.decoder.model import load_decoder

    model, tokenizer = load_decoder(cfg, for_training=False)
    model = PeftModel.from_pretrained(model, cfg.infer.checkpoint)
    model.eval()
    return model, tokenizer


def main():
    cfg = get_config()
    mode = cfg.get("eval_mode", "test")
    assert cfg.infer.checkpoint, "pass infer.checkpoint=<trained LoRA adapter dir>"

    if cfg.decoder.modality == "vision":
        raise NotImplementedError(
            "vision-modality evaluation needs image-aware generation "
            "(pss.decoder.generate is text-only so far) — see "
            "pss/decoder/train.py's matching NotImplementedError note."
        )

    model, tokenizer = load_eval_model(cfg)
    result = collect_predictions(cfg, model, tokenizer, mode)
    all_pred, all_true = result["all_pred"], result["all_true"]
    streams, type_pairs = result["streams"], result["type_pairs"]

    n_types = len(read_class_names(cfg.data.root))
    prf = metrics.prf_from_preds(all_pred, all_true)
    kappa = metrics.kappa_from_preds(all_pred, all_true)
    sm = metrics.stream_metrics(streams)
    type_f1 = metrics.type_macro_f1(type_pairs, n_types)

    print(
        f"\n=== PSS decoder eval ({mode}, modality={cfg.decoder.modality}, "
        f"context_mode={cfg.decoder.context_mode}) ==="
    )
    print(
        f"boundary  P={prf['precision']:.4f}  R={prf['recall']:.4f}  "
        f"F1={prf['f1']:.4f}  kappa={kappa:.4f}"
    )
    print(f"stream    MNDD={sm['mndd']:.4f}  STP={sm['stp']:.4f}")
    print(f"typing    doc macro-F1={type_f1:.4f}")


if __name__ == "__main__":
    main()
