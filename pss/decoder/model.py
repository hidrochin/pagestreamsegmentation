"""
Loads the decoder base model + tokenizer/processor via Unsloth and applies a
LoRA adapter, dispatching on ``decoder.modality``. Requires the ``.venv-decoder``
environment (see ``requirements-decoder.txt``) — Unsloth is CUDA-only, so this
module cannot be exercised on a CPU/MPS-only machine; only import it on the
target GPU box.
"""


def load_decoder(cfg, for_training=True):
    """Returns (model, tokenizer_or_processor). ``model`` already has the LoRA
    adapter attached when for_training=True; pass for_training=False to load a
    previously-exported merged checkpoint for inference (no adapter wrapping)."""
    if cfg.decoder.modality == "vision":
        return _load_vision(cfg, for_training)
    return _load_text(cfg, for_training)


def _load_text(cfg, for_training):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.decoder.base_model,
        max_seq_length=cfg.decoder.max_seq_length,
        load_in_4bit=cfg.decoder.load_in_4bit,
    )
    if for_training:
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.decoder.lora.r,
            lora_alpha=cfg.decoder.lora.alpha,
            lora_dropout=cfg.decoder.lora.dropout,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=cfg.seed,
        )
    return model, tokenizer


def _load_vision(cfg, for_training):
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        model_name=cfg.decoder.base_model,
        load_in_4bit=cfg.decoder.load_in_4bit,
    )
    if for_training:
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=cfg.decoder.lora.r,
            lora_alpha=cfg.decoder.lora.alpha,
            lora_dropout=cfg.decoder.lora.dropout,
            bias="none",
            random_state=cfg.seed,
        )
    return model, processor
