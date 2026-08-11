"""
Fine-tunes the decoder PSS model with Unsloth + TRL's SFTTrainer, loss computed
on completion tokens only (matching the source paper, arXiv:2408.11981 Sec B.3).
Requires the ``.venv-decoder`` environment on a CUDA GPU — Unsloth has no
CPU/MPS backend, so this cannot run on the Mac dev machine; see
``pss/decoder/README.md`` for offline staging to the target L40 box.

    python -m pss.decoder.train --config=configs/pss_decoder.yaml
    python -m pss.decoder.train --config=configs/pss_decoder.yaml decoder.context_mode=pair

Checkpoints (LoRA adapter only) land under ``{workspace}/checkpoints/{run_id}/``,
``run_id`` defaulting to a timestamp (``pss/decoder/config.py::_derive``) so
repeated runs never clobber each other's checkpoints — mirrors the encoder
pipeline's convention (``pss/train.py``). Run ``pss.decoder.export`` afterwards
to merge the adapter into a standalone HF model dir for production serving.

Vision-modality (``decoder.modality=vision``) training needs Unsloth's
vision-SFT data collator (image + text message batches), which differs enough
from the text path that it isn't implemented here — see the NotImplementedError
below for the extension point.
"""

import os

from pss.decoder.config import get_config
from pss.decoder.dataset import build_cache, load_cache
from pss.decoder.model import load_decoder


def _to_hf_dataset(examples, tokenizer):
    """List[{"prompt","completion","images"}] -> a datasets.Dataset with a
    single "text" column: the base model's own chat template applied to a
    [user: prompt, assistant: completion] turn pair. Using the model's real
    chat template (not a hand-rolled marker) keeps the completion-only
    collator's response_template match exact."""
    from datasets import Dataset

    texts = [
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": ex["prompt"]},
                {"role": "assistant", "content": ex["completion"]},
            ],
            tokenize=False,
        )
        for ex in examples
    ]
    return Dataset.from_dict({"text": texts})


def main():
    cfg = get_config()
    print(f"[decoder.train] run_id={cfg.run_id}  checkpoints -> {cfg.save_weight_dir}")
    os.makedirs(cfg.save_weight_dir, exist_ok=True)

    if cfg.decoder.modality == "vision":
        raise NotImplementedError(
            "decoder.modality=vision training needs Unsloth's vision SFT data "
            "collator (image + text message batches) — see FastVisionModel's "
            "training docs and pss/decoder/model.py::_load_vision (already "
            "wired for loading + LoRA); only the SFTTrainer data path here is "
            "text-only so far. pss.decoder.dataset already emits per-example "
            "image paths (ex['images']) ready to feed into that collator."
        )

    train_path = build_cache(cfg, "train")
    val_path = build_cache(cfg, "val")
    train_examples = load_cache(train_path)
    val_examples = load_cache(val_path)
    print(f"[decoder.train] {len(train_examples)} train / {len(val_examples)} val examples")

    model, tokenizer = load_decoder(cfg, for_training=True)

    from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

    train_ds = _to_hf_dataset(train_examples, tokenizer)
    val_ds = _to_hf_dataset(val_examples, tokenizer)

    collator = DataCollatorForCompletionOnlyLM(
        response_template=cfg.decoder.chat_response_template, tokenizer=tokenizer
    )

    args = SFTConfig(
        output_dir=cfg.save_weight_dir,
        per_device_train_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=cfg.train.lr,
        max_steps=cfg.train.max_steps,
        warmup_steps=cfg.train.warmup_steps,
        weight_decay=cfg.train.weight_decay,
        optim=cfg.train.optimizer,
        save_steps=cfg.train.save_every,
        eval_steps=cfg.train.eval_every,
        eval_strategy="steps",
        logging_steps=50,
        bf16=True,
        seed=cfg.seed,
        dataset_text_field="text",
        max_seq_length=cfg.decoder.max_seq_length,
        report_to="tensorboard",
        dataloader_num_workers=cfg.train.num_workers,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=args,
        data_collator=collator,
    )
    trainer.train()

    final_dir = os.path.join(cfg.save_weight_dir, "last")
    trainer.save_model(final_dir)
    print(f"[decoder.train] saved final LoRA adapter -> {final_dir}")


if __name__ == "__main__":
    main()
