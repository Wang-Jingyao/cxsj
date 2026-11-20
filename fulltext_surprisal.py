"""Compute full-text token surprisal with overlapping windows (Qwen3-4B-Base).

Example:
  .venv/bin/python -B fulltext_surprisal.py --text /nas/data/gutenberg/corpus/84.txt
Each token after the first is scored exactly once. No text or model files are modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
HOME = Path("/home/cxsj26f-g4").resolve()


def inside_home(path):
    path = Path(path).resolve()
    if not path.is_relative_to(HOME):
        raise ValueError(f"Writable path must be inside {HOME}: {path}")
    return path


def windows(length, size, stride):
    """Yield (input_start, input_end, score_start); end is exclusive."""
    end = min(size, length)
    yield 0, end, 1
    previous_end = end
    while previous_end < length:
        end = min(previous_end + stride, length)
        start = max(0, end - size)
        yield start, end, previous_end
        previous_end = end


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/data1/model/qwen3-4b-base"))
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--window", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--encoding", default="utf-8-sig")
    args = parser.parse_args()
    if not 1 <= args.stride < args.window:
        parser.error("Require 1 <= stride < window")
    output = inside_home(args.output_dir or HOME / "data" / "gutenberg" /
                         (args.text.stem + "_qwen3_4b_full"))
    # Refuse to overwrite previous results, or follow output symlinks outside HOME.
    targets = [inside_home(output / name) for name in
               ("surprisal.npy", "token_ids.npy", "metadata.json", "windows.jsonl")]
    if any(p.exists() for p in targets):
        parser.error("Output files already exist; choose a new --output-dir")
    cache = inside_home(ROOT / ".cache")
    cache.mkdir(exist_ok=True)
    os.environ["HF_HOME"] = str(inside_home(cache / "huggingface"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TORCH_HOME"] = str(inside_home(cache / "torch"))
    os.environ["TMPDIR"] = str(cache)
    import numpy as np
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM

    source = args.text.resolve(strict=True)
    raw = source.read_bytes()
    text = raw.decode(args.encoding)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    # Preserve all text, including Gutenberg headers/footers. No chat template or BOS.
    ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    count = len(ids)
    if count < 2:
        parser.error("Input needs at least two tokens")
    print(f"Source: {source}\nCharacters: {len(text)}; tokens: {count}", flush=True)
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype,
        attn_implementation="sdpa",
    ).to(args.device).eval()
    if args.window > model.config.max_position_embeddings:
        parser.error("Window exceeds the model's configured context limit")
    schedule = list(windows(count, args.window, args.stride))
    scores = np.empty(count - 1, dtype=np.float32)
    covered = np.zeros(count - 1, dtype=np.uint8)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with targets[3].open("x", encoding="utf-8") as log, torch.inference_mode():
        for index, (start, end, score_start) in enumerate(schedule, 1):
            batch = torch.tensor([ids[start:end]], dtype=torch.long, device=args.device)
            out = model(input_ids=batch, use_cache=False)
            first = score_start - start
            # Local logit at j-1 predicts token j. Score only newly exposed tokens.
            losses = F.cross_entropy(
                out.logits[0, first - 1:-1].float(),
                batch[0, first:], reduction="none",
            ).cpu().numpy()
            dest = slice(score_start - 1, end - 1)
            if losses.size != end - score_start or covered[dest].any():
                raise RuntimeError("Window alignment or duplicate scoring error")
            scores[dest] = losses
            covered[dest] += 1
            log.write(json.dumps(dict(window=index, input_start=start, input_end=end,
                                      score_start=score_start, score_end=end)) + "\n")
            del out, batch, losses
            if index == 1 or index % 10 == 0 or index == len(schedule):
                print(f"Window {index}/{len(schedule)}; scored through token {end - 1}; "
                      f"elapsed {time.monotonic() - started:.1f}s", flush=True)
    if not np.all(covered == 1) or not np.isfinite(scores).all():
        raise RuntimeError("Coverage/finite-value validation failed")
    with targets[0].open("xb") as handle:
        np.save(handle, scores, allow_pickle=False)
    with targets[1].open("xb") as handle:
        np.save(handle, np.asarray(ids, dtype=np.int64), allow_pickle=False)
    metadata = dict(
        source=str(source), source_sha256=hashlib.sha256(raw).hexdigest(),
        characters=len(text), model=str(args.model.resolve()), dtype=str(dtype),
        window=args.window, stride=args.stride, windows=len(schedule),
        token_count=count, surprisal_count=len(scores),
        token_index_start=1, token_index_end=count - 1,
        coverage_exactly_once=bool(np.all(covered == 1)),
        unit="nats", add_special_tokens=False, text_truncated=False,
        cleaning="none; source headers and footers retained",
        context_policy="overlapping windows; only new target tokens scored",
        elapsed_seconds=time.monotonic() - started,
        mean_surprisal=float(scores.mean()),
    )
    with targets[2].open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    print(f"Saved results to {output}", flush=True)


if __name__ == "__main__":
    main()
