"""Plot surprisal, power spectrum and ACF using the existing AutoPeriod implementation.

Run: .venv/bin/python -B plot_periodicity.py
Or:  .venv/bin/python -B plot_periodicity.py --nll wsj_2380_nll.npy
"""
import argparse
import json
import hashlib
import time
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
HOME = Path("/home/cxsj26f-g4").resolve()


def writable_path(path):
    path = path.resolve()
    if not path.is_relative_to(HOME):
        raise ValueError(f"Output must remain inside {HOME}: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", type=Path, help="UTF-8 text; default: test/t1.txt")
    source.add_argument("--nll", type=Path, help="Existing one-dimensional NLL .npy")
    parser.add_argument("--model", default="/data1/model/gpt2-small")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    if args.iterations < 1 or not 0 < args.confidence < 1:
        parser.error("iterations must be positive; confidence must be between 0 and 1")
    if args.max_length < 8:
        parser.error("max-length must be at least 8")

    # Keep caches local and use a headless plotting backend.
    os.environ["MPLCONFIGDIR"] = str(writable_path(ROOT / ".mplconfig"))
    os.environ["HF_HOME"] = str(writable_path(ROOT / ".cache" / "huggingface"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import matplotlib
    except ImportError:
        parser.exit(1, "Missing matplotlib. Install with: .venv/bin/python -m pip install --no-cache-dir matplotlib\n")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from autoperiod import AutoPeriod

    output = writable_path(args.output_dir)
    if args.nll:
        values = np.load(args.nll, allow_pickle=False)
    else:
        from nll import NLLCal
        text_path = args.text or ROOT / "test" / "t1.txt"
        text = text_path.read_text(encoding="utf-8").strip()
        if not text:
            parser.error("Text input is empty")
        calculator = NLLCal(args.model, device=args.device, batch_size=1,
                            max_length=args.max_length)
        calculator.model.eval()
        token_count = len(calculator.tokenizer.encode(text))
        if token_count > args.max_length:
            print(f"WARNING: {token_count} tokens; only the first {args.max_length} are analyzed.")
        values = calculator.texts_to_nll([text])[0].float().cpu().numpy()

    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 8 or not np.all(np.isfinite(values)):
        parser.error("NLL must be a finite one-dimensional array with at least 8 values")
    if np.isclose(np.var(values), 0):
        parser.error("NLL has no usable variation")
    np.random.seed(args.seed)
    times = np.arange(values.size)
    detector = AutoPeriod(times, values.copy(), mc_iterations=args.iterations,
                          confidence_level=args.confidence)
    started = time.monotonic()
    print("Running APS: " + str(values.size) + " values, " + str(args.iterations) + " shuffles", flush=True)
    results = detector.run()
    print(results)
    print(f"Analyzed {values.size} surprisal values; seed={args.seed}")
    output.mkdir(parents=True, exist_ok=True)

    report = {
        "periods": sorted(float(p) for p in results["periods"]),
        "period_hints": sorted(float(p) for p in results["period_hints"]),
        "threshold": float(results["threshold"]),
        "period_unit": "tokens",
        "surprisal_count": int(values.size),
        "iterations": args.iterations,
        "confidence_level": args.confidence,
        "seed": args.seed,
        "fft": "lomb_scargle",
        "acf_method": "unbiased",
        "elapsed_seconds": time.monotonic() - started,
    }
    if args.nll:
        report["input_nll"] = str(args.nll.resolve())
        report["input_sha256"] = hashlib.sha256(args.nll.read_bytes()).hexdigest()
        metadata_path = args.nll.parent / "metadata.json"
        if metadata_path.is_file():
            report["input_metadata"] = json.loads(metadata_path.read_text())
    else:
        report["input_text"] = str(text_path.resolve())
        report["model"] = args.model
        report["max_length"] = args.max_length
    result_path = writable_path(output / "aps_results.json")
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results saved to {result_path}", flush=True)

    def save(fig, name):
        target = writable_path(output / name)
        fig.tight_layout()
        fig.savefig(target, dpi=180)
        plt.close(fig)
        print(target)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, values, lw=0.8)
    ax.set(xlabel="Token position (first predicted token = 0)",
           ylabel="Surprisal (nats)", title="Token surprisal")
    ax.grid(alpha=0.25)
    save(fig, "01_surprisal.png")

    mask = (detector.periods > 2) & (detector.periods < detector.time_span / 2)
    periods = detector.periods[mask]
    order = np.argsort(periods)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(periods[order], detector.powers[mask][order], label="Observed spectrum")
    ax.axhline(detector.power_threshold, color="red", ls="--",
               label=f"Shuffle max-power threshold ({args.confidence:.0%})")
    for i, period in enumerate(sorted(results["periods"])):
        ax.axvline(period, color="green", ls=":", alpha=0.6,
                   label="ACF-validated period" if i == 0 else None)
    ax.set(xlabel="Period (tokens)", ylabel="Power", title="Surprisal power spectrum")
    ax.legend()
    ax.grid(alpha=0.25)
    save(fig, "02_spectrum.png")

    max_lag = values.size // 2
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(np.arange(max_lag + 1), detector.acf[:max_lag + 1])
    ax.axhline(0, color="gray", lw=0.8)
    for period in sorted(results["periods"]):
        ax.axvline(period, color="green", ls=":", alpha=0.6)
        ax.text(period, 0.95, f"T={period:g}", transform=ax.get_xaxis_transform(),
                rotation=90, va="top", ha="right", fontsize=8)
    ax.set(xlabel="Lag (tokens)", ylabel="Autocorrelation",
           title="Surprisal ACF (first half of lags)")
    ax.grid(alpha=0.25)
    save(fig, "03_acf.png")


if __name__ == "__main__":
    main()
