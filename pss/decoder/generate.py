"""Batched text generation, shared by ``pss.decoder.evaluate`` (scores against
ground truth) and ``pss.decoder.infer`` (production, no labels). Text-modality
only for now — see ``pss/decoder/train.py``'s vision NotImplementedError note;
extend here alongside that when vision inference is wired up."""


def generate_completions(cfg, model, tokenizer, prompts):
    """prompts: list[str], each already run through the model's chat template
    with add_generation_prompt=True. Returns list[str] raw completions
    (decoded new tokens only, special tokens stripped), same order/length."""
    import torch

    tokenizer.padding_side = "left"  # decoder-only batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    outs = []
    bs = cfg.infer.batch_size
    for i in range(0, len(prompts), bs):
        batch = prompts[i : i + bs]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=cfg.infer.max_new_tokens, do_sample=False
            )
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        outs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return outs


def build_prompts(tokenizer, examples):
    """[{"prompt": ...}] -> chat-templated prompt strings ready for generation."""
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for ex in examples
    ]
