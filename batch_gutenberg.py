"""Resume-safe 20 Chinese + 20 English Gutenberg batch; GPUs 0/1, 4 APS CPU workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
HOME = Path("/home/cxsj26f-g4").resolve()
BATCH = HOME / "data/gutenberg/batch_20zh_20en_20261109"
ORIGINAL = ("example.py", "nll.py", "autoperiod.py")
LOCK = threading.Lock()


def safe(path):
    path = Path(path).resolve()
    if not path.is_relative_to(HOME):
        raise ValueError(f"Write outside authorized home: {path}")
    return path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = safe(path)
    temp = safe(path.with_suffix(path.suffix + ".tmp"))
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def prepare():
    manifest_path = BATCH / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/data1/model/qwen3-4b-base",
                                               local_files_only=True)
    with (BATCH / "catalog.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    samples = []
    for language in ("zh", "en"):
        candidates = [r for r in rows if r["Type"] == "Text" and
                      r["Language"].strip() == language]
        candidates.sort(key=lambda r: int(r["Text#"]))
        random.Random(42).shuffle(candidates)
        chosen = []
        seen_titles = set()
        seen_hashes = set()
        for row in candidates:
            title_key = " ".join(row["Title"].split()).casefold()
            if title_key in seen_titles:
                continue
            source = Path("/nas/data/gutenberg/corpus") / (row["Text#"] + ".txt")
            try:
                raw = source.read_bytes()
                text = raw.decode("utf-8-sig")
            except (OSError, UnicodeError):
                continue
            source_hash = hashlib.sha256(raw).hexdigest()
            if source_hash in seen_hashes:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
            if not 4096 <= len(ids) <= 32768:
                continue
            sample = dict(id=row["Text#"], language=language, title=row["Title"],
                          subjects=row["Subjects"], source=str(source),
                          source_sha256=hashlib.sha256(raw).hexdigest(),
                          token_count=len(ids), characters=len(text))
            chosen.append(sample)
            seen_titles.add(title_key)
            seen_hashes.add(source_hash)
            print(f"Selected {language} {len(chosen)}/20: {sample['id']} "
                  f"{len(ids)} tokens {sample['title']}", flush=True)
            if len(chosen) == 20:
                break
        if len(chosen) != 20:
            raise RuntimeError(f"Only {len(chosen)} eligible {language} samples")
        samples.extend(chosen)
    manifest = dict(
        samples=samples, seed=42, count_per_language=20,
        selection="Seeded shuffle of numeric-ID-sorted monolingual Text catalog rows; "
                  "first 20 readable UTF-8 whole catalog documents per language with 4096..32768 tokens; exact normalized titles and source hashes deduplicated",
        catalog_sha256=digest(BATCH / "catalog.csv"),
        catalog_source="https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv",
        catalog_note="Previously downloaded local catalog snapshot; not refreshed for this run",
        model="/data1/model/qwen3-4b-base", window=2048, stride=512,
        iterations=1000, confidence_level=.99, aps_seed=42,
        original_hashes={name:digest(ROOT / name) for name in ORIGINAL},
    )
    write_json(manifest_path, manifest)
    return manifest


def sample_dir(sample):
    return safe(BATCH / (sample["language"] + "_" + sample["id"]))


def valid_nll(sample):
    import numpy as np
    directory = sample_dir(sample)
    try:
        m=json.loads((directory / "metadata.json").read_text())
        v=np.load(directory / "surprisal.npy", allow_pickle=False)
        return (m["source_sha256"] == sample["source_sha256"]
                and m["model"] == "/data1/model/qwen3-4b-base"
                and m["window"] == 2048 and m["stride"] == 512
                and m["coverage_exactly_once"] and not m["text_truncated"]
                and v.shape == (sample["token_count"]-1,) and np.isfinite(v).all())
    except (OSError, ValueError, KeyError):
        return False


def valid_result(sample):
    directory=sample_dir(sample)
    try:
        r=json.loads((directory / "aps" / "aps_results.json").read_text())
        return (valid_nll(sample) and r["input_sha256"] == digest(directory / "surprisal.npy")
                and r["iterations"] == 1000 and r["confidence_level"] == .99
                and r["seed"] == 42 and r["surprisal_count"] == sample["token_count"]-1
                and all((directory / "aps" / name).is_file() for name in
                        ("01_surprisal.png","02_spectrum.png","03_acf.png")))
    except (OSError, ValueError, KeyError):
        return False


def run_command(command, log_path):
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1")
    with safe(log_path).open("a", encoding="utf-8") as log:
        result=subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                              stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}; see {log_path}")


def run_aps(sample):
    directory=sample_dir(sample)
    print(f"APS start: {sample['language']}_{sample['id']}", flush=True)
    run_command([sys.executable,"-u","-B",str(ROOT/"plot_periodicity.py"),
                 "--nll",str(directory/"surprisal.npy"),"--iterations","1000",
                 "--confidence","0.99","--seed","42","--output-dir",str(directory/"aps")],
                directory/"aps.log")
    if not valid_result(sample):
        raise RuntimeError("APS result validation failed")
    print(f"APS done: {sample['language']}_{sample['id']}", flush=True)


def run_group(samples, device, aps_pool, submitted, errors):
    for sample in samples:
        key=sample["language"]+"_"+sample["id"]
        try:
            if valid_result(sample):
                print("Already complete: "+key, flush=True)
                continue
            directory=sample_dir(sample)
            directory.mkdir(parents=True,exist_ok=True)
            if not valid_nll(sample):
                if digest(sample["source"]) != sample["source_sha256"]:
                    raise RuntimeError("Source changed since sample selection")
                # Preserve partial output before retrying a failed inference.
                leftovers=[directory/name for name in
                           ("surprisal.npy","token_ids.npy","metadata.json","windows.jsonl")]
                if any(p.exists() for p in leftovers):
                    archive=safe(directory / ("partial_"+str(time.time_ns())))
                    archive.mkdir()
                    for p in leftovers:
                        if p.exists():
                            safe(p).rename(safe(archive/p.name))
                print(f"NLL start {key} on {device}",flush=True)
                run_command([sys.executable,"-u","-B",str(ROOT/"fulltext_surprisal.py"),
                             "--text",sample["source"],"--device",device,
                             "--window","2048","--stride","512",
                             "--output-dir",str(directory)],directory/"nll.log")
                if not valid_nll(sample):
                    raise RuntimeError("NLL coverage validation failed")
                print(f"NLL done {key} on {device}",flush=True)
            future=aps_pool.submit(run_aps,sample)
            with LOCK:
                submitted.append((sample,future))
        except Exception as exc:
            with LOCK:
                errors[key]=str(exc)
            print(f"ERROR {key}: {exc}",flush=True)


def summarize(manifest, errors):
    rows=[]
    for sample in manifest["samples"]:
        key=sample["language"]+"_"+sample["id"]
        row=dict(language=sample["language"],id=sample["id"],title=sample["title"],
                 token_count=sample["token_count"],status="pending",
                 periods=[],period_hints=[],result_file=str(sample_dir(sample)/"aps/aps_results.json"))
        if valid_result(sample):
            r=json.loads(Path(row["result_file"]).read_text())
            row.update(status="complete",periods=r["periods"],period_hints=r["period_hints"],
                       threshold=r["threshold"],aps_seconds=r["elapsed_seconds"])
        elif key in errors:
            row.update(status="failed",error=errors[key])
        elif valid_nll(sample):
            row["status"]="aps_pending"
        rows.append(row)
    write_json(BATCH/"summary.json",rows)
    target=safe(BATCH/"summary.csv")
    with target.open("w",encoding="utf-8-sig",newline="") as f:
        fields=["language","id","title","token_count","status","periods","period_hints",
                "threshold","aps_seconds","result_file","error"]
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row,"periods":json.dumps(row["periods"]),
                             "period_hints":json.dumps(row["period_hints"])})
    counts={lang:sum(r["status"]=="complete" and r["language"]==lang for r in rows)
            for lang in ("zh","en")}
    write_json(BATCH/"status.json",dict(completed=counts,errors=errors,
                                      updated_unix=time.time(),pid=os.getpid()))
    print(f"Completed: zh={counts['zh']}/20 en={counts['en']}/20",flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    safe(BATCH).mkdir(parents=True,exist_ok=True)
    cache=safe(ROOT/".cache")
    cache.mkdir(exist_ok=True)
    os.environ.update(HF_HOME=str(safe(cache/"huggingface")),HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1",TMPDIR=str(cache))
    manifest=prepare()
    if args.prepare_only:
        return
    # An OS lock prevents duplicate batch launches, and is released on exit.
    import fcntl
    with safe(BATCH/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write_json(BATCH/"run.json",dict(pid=os.getpid(),started_unix=time.time(),
                                       devices=["cuda:0","cuda:1"]))
        errors={}
        submitted=[]
        summarize(manifest,errors)
        with ThreadPoolExecutor(max_workers=4) as aps_pool:
            with ThreadPoolExecutor(max_workers=2) as gpu_pool:
                futures=[gpu_pool.submit(run_group,manifest["samples"][i::2],
                                         f"cuda:{i}",aps_pool,submitted,errors)
                         for i in range(2)]
                for future in as_completed(futures):
                    future.result()
                    summarize(manifest,errors)
            by_future={future:sample for sample,future in submitted}
            for future in as_completed(by_future):
                sample=by_future[future]
                try:
                    future.result()
                except Exception as exc:
                    errors[sample["language"]+"_"+sample["id"]]=str(exc)
                summarize(manifest,errors)
        rows=summarize(manifest,errors)
        unchanged=all(digest(ROOT/name)==value for name,value in manifest["original_hashes"].items())
        complete=(len(rows)==40 and all(r["status"]=="complete" for r in rows) and unchanged)
        write_json(BATCH/"completion.json",dict(complete=complete,original_files_unchanged=unchanged,
                                               completed_unix=time.time()))
        if not complete:
            raise SystemExit(1)


if __name__=="__main__":
    main()
