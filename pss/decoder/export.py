"""
Merges a trained LoRA adapter into its base model and saves a standalone HF
model dir (safetensors) — the decoder analogue of ``pss/export_pth.py``. Run
this before serving in production; the merged dir loads with plain
``transformers.AutoModelForCausalLM.from_pretrained(...)`` + ``AutoTokenizer``,
no PEFT or Unsloth dependency needed at serve time.

    python -m pss.decoder.export --config=configs/pss_decoder.yaml \
        infer.checkpoint=pss_runs/decoder_exp/checkpoints/<run_id>/last

Writes to ``{infer.checkpoint}/merged`` by default — override with ``export.out``.
Requires ``.venv-decoder``.
"""

import os

from pss.decoder.config import get_config


def merge_and_save(cfg):
    from peft import PeftModel

    from pss.decoder.model import load_decoder

    model, tokenizer = load_decoder(cfg, for_training=False)
    model = PeftModel.from_pretrained(model, cfg.infer.checkpoint)
    model = model.merge_and_unload()

    out = cfg.export.out or os.path.join(cfg.infer.checkpoint, "merged")
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    return out


def main():
    cfg = get_config()
    assert cfg.infer.checkpoint, "pass infer.checkpoint=<trained LoRA adapter dir>"
    out = merge_and_save(cfg)
    print(f"[decoder.export] {cfg.infer.checkpoint} -> {out}")


if __name__ == "__main__":
    main()
