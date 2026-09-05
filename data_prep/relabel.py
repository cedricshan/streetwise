"""Relabels clips with a cloud vision-language model into the JSON alert schema."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import importlib.util
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
RESPONSES_URL = "https://api.openai.com/v1/responses"
DEFAULT_PROMPT = Path("runs/relabel/prompts/v1sp6_2026-08-11.py")
DEFAULT_MANIFEST = Path("runs/relabel/splits_2026-08-10/train.json")

EXPECTED_CACHED_TOKENS = 1102
CACHE_MIN_TOKENS = 1024

def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

def load_prompt(path: Path, name: str) -> str:
    spec = importlib.util.spec_from_file_location("v1sp5_prompt", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)

def stratum(row: dict) -> str:

    src = row["source"]
    if src == "wad":
        return f"wad:{row['orig'].get('danger_level')}"
    if src == "wrd":
        return f"wrd:{row['orig'].get('category')}"
    if src == "egoblind":
        return f"egoblind:{row.get('task')}"
    return src

def stratified_sample(rows: list[dict], per_source: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)

    picked: list[dict] = []
    for src in sorted(by_source):
        buckets: dict[str, list[dict]] = defaultdict(list)
        for r in by_source[src]:
            buckets[stratum(r)].append(r)

        sizes = {k: len(v) for k, v in buckets.items()}
        total = sum(sizes.values())
        raw = {k: per_source * n / total for k, n in sizes.items()}
        quota = {k: int(v) for k, v in raw.items()}
        left = per_source - sum(quota.values())
        for k in sorted(raw, key=lambda k: raw[k] - quota[k], reverse=True)[:left]:
            quota[k] += 1
        for k in sorted(buckets):
            lst = list(buckets[k])
            rng.shuffle(lst)
            picked += lst[: quota[k]]
    return picked

def encode_frame(path: Path, longest_edge: int, quality: int) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((longest_edge, longest_edge))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def build_content(prompt: str, frames_dir: Path, args) -> list[dict]:

    content: list[dict] = [
        {
            "type": "input_text",
            "text": prompt,
            "prompt_cache_breakpoint": {"mode": "explicit"},
        }
    ]
    for k in range(9):
        tag = f"Frame {k + 1} of 9" + (" - THE PRESENT MOMENT" if k == 8 else "")
        edge = args.final_edge if k == 8 else args.context_edge
        content.append({"type": "input_text", "text": tag})
        content.append(
            {
                "type": "input_image",
                "image_url": encode_frame(frames_dir / f"{k}.jpg", edge, args.jpeg_quality),
            }
        )
    return content

def build_body(prompt: str, row: dict, shard: int, args, service_tier: str) -> dict:
    return {
        "model": args.model,
        "service_tier": service_tier,
        "prompt_cache_key": f"{args.cache_key_prefix}-{shard:02d}",
        "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
        "reasoning": {"effort": args.reasoning_effort},
        "max_output_tokens": args.max_output_tokens,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": build_content(prompt, Path(row["frames_dir"]), args),
            }
        ],
    }

def post(body: dict, api_key: str, timeout: int) -> tuple[dict | None, str | None]:
    req = urllib.request.Request(
        RESPONSES_URL,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read()), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode()[:400]}"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

def response_text(payload: dict) -> str:

    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    chunks: list[str] = []
    for item in payload.get("output") or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") in ("output_text", "text") and part.get("text"):
                chunks.append(part["text"])
    return "".join(chunks)

REQUIRED_FIELDS = ("alert", "hazards", "onset", "user_state", "scene", "danger_level")

def parse_label(text: str) -> tuple[dict | None, str]:

    depth = 0
    start = -1
    candidates: list[str] = []
    for i, ch in enumerate(text or ""):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start : i + 1])
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        missing = [f for f in REQUIRED_FIELDS if f not in obj]
        if missing:
            return None, f"missing fields: {missing}"
        if not isinstance(obj.get("danger_level"), int) or not 1 <= obj["danger_level"] <= 5:
            return None, f"bad danger_level: {obj.get('danger_level')!r}"
        if not isinstance(obj.get("hazards"), list):
            return None, "hazards is not a list"
        return obj, ""
    return None, "no parseable JSON object"

def shard_of(clip_id: str, shards: int) -> int:
    return int(hashlib.md5(clip_id.encode()).hexdigest(), 16) % shards

def done_ids(out_dir: Path) -> set[str]:
    ids: set[str] = set()
    for f in sorted(out_dir.glob("shard-*.jsonl")):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("parse_ok") or rec.get("raw"):
                ids.add(rec["id"])
    return ids

class Monitor:

    def __init__(self, strict: bool, expect_cached: int):
        self.lock = threading.Lock()
        self.strict = strict
        self.expect = expect_cached
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.post_warmup = 0
        self.abort_reason: str | None = None
        self.credit_reason: str | None = None

    def credit_error(self, err: str) -> None:

        with self.lock:
            if self.credit_reason is None:
                self.credit_reason = err[:200]

    def record(self, cached: int | None, write: int | None, warmup: bool) -> None:
        with self.lock:
            if (write or 0) > 0 and not warmup:
                self.writes += 1
                if self.writes >= 3 and self.abort_reason is None:
                    self.abort_reason = (
                        f"cache_write_tokens>0 on {self.writes} post-warm-up requests — "
                        "prompt_cache_options.mode is not taking effect; every request is "
                        "being billed a 1.25x cache write on the image tokens"
                    )
            if warmup:
                return
            self.post_warmup += 1
            if (cached or 0) >= CACHE_MIN_TOKENS:
                self.hits += 1
            else:
                self.misses += 1
                if self.post_warmup >= 10 and self.hits == 0 and self.abort_reason is None:
                    self.abort_reason = (
                        f"{self.misses} post-warm-up requests with cached_tokens<{CACHE_MIN_TOKENS} "
                        "and zero hits — the prefix is not being cached (silent-failure mode)"
                    )

    def should_abort(self) -> str | None:
        if self.credit_reason:
            return f"API credits exhausted: {self.credit_reason}"
        return self.abort_reason if self.strict else None

def label_one(row: dict, prompt: str, args, api_key: str, monitor: Monitor, warmup: bool,
              emit) -> dict:

    shard = shard_of(row["id"], args.shards)
    tier = args.service_tier
    attempts = 0
    err = None
    t0 = time.time()
    payload = None
    while attempts < args.retries:
        attempts += 1
        try:
            body = build_body(prompt, row, shard, args, tier)
        except OSError as e:

            err = f"local: {type(e).__name__}: {e}"
            break
        payload, err = post(body, api_key, args.timeout)
        if payload is not None:
            break
        if err and ("no credits" in err.lower() or "insufficient_quota" in err):
            monitor.credit_error(err)
            break
        if monitor.should_abort():
            break
        transient = err and ("HTTP 429" in err or "HTTP 5" in err or "timed out" in err.lower())
        if not transient:
            break

        if tier == "flex" and attempts >= args.flex_fallback_after:
            tier = "auto"
        time.sleep(min(args.backoff * (2 ** (attempts - 1)), 30))

    rec: dict = {
        "id": row["id"],
        "source": row["source"],
        "frames_dir": row["frames_dir"],
        "model": args.model,
        "prompt_version": args.prompt_version,
        "prompt_sha1": hashlib.sha1(prompt.encode()).hexdigest(),
        "shard": shard,
        "cache_key": f"{args.cache_key_prefix}-{shard:02d}",
        "service_tier": tier,
        "context_edge": args.context_edge,
        "final_edge": args.final_edge,
        "warmup": warmup,
        "attempts": attempts,
        "latency_s": round(time.time() - t0, 2),
        "ts": round(time.time(), 3),
    }

    if payload is None:
        rec.update(error=err, raw=None, parsed=None, parse_ok=False)
        emit(rec)
        return rec

    usage = payload.get("usage") or {}
    details = usage.get("input_tokens_details") or {}
    text = response_text(payload)
    parsed, parse_err = parse_label(text)
    rec.update(
        raw=text,
        parsed=parsed,
        parse_ok=parsed is not None,
        parse_error=parse_err or None,
        usage=usage,
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        cached_tokens=details.get("cached_tokens"),
        cache_write_tokens=details.get("cache_write_tokens"),
        status=payload.get("status"),
        error=None,
    )
    monitor.record(rec["cached_tokens"], rec["cache_write_tokens"], warmup)
    emit(rec)
    return rec

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT)
    ap.add_argument("--prompt-name", default="V1SP6")
    ap.add_argument("--prompt-version", default="v1sp6_2026-08-11")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--ids-file", type=Path, help="one manifest id per line")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--sample-per-source", type=int, default=0,
                    help="write a stratified id list and exit")
    ap.add_argument("--sample-out", type=Path)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--model", default="gpt-5.6-terra")
    ap.add_argument("--service-tier", default="flex", choices=["flex", "auto", "default", "priority"])
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--max-output-tokens", type=int, default=300)
    ap.add_argument("--context-edge", type=int, default=448)
    ap.add_argument("--final-edge", type=int, default=768)
    ap.add_argument("--jpeg-quality", type=int, default=82)
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--cache-key-prefix", default="streetwise-v1sp6")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--backoff", type=float, default=3.0)
    ap.add_argument("--flex-fallback-after", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--strict", action="store_true",
                    help="abort the run if the caching contract is violated")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    load_env(REPO_ROOT / ".env")
    rows = json.loads(args.manifest.read_text())
    by_id = {r["id"]: r for r in rows}

    if args.sample_per_source:
        picked = stratified_sample(rows, args.sample_per_source, args.seed)
        out = args.sample_out or (args.out_dir / "ids.txt" if args.out_dir else None)
        if out is None:
            ap.error("--sample-per-source needs --sample-out or --out-dir")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(r["id"] for r in picked) + "\n")
        counts = Counter(stratum(r) for r in picked)
        print(f"wrote {len(picked)} ids -> {out}")
        for k in sorted(counts):
            print(f"  {k:28s} {counts[k]}")
        return

    if not args.out_dir:
        ap.error("--out-dir is required")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.ids_file:
        wanted = [i.strip() for i in args.ids_file.read_text().splitlines() if i.strip()]
        missing = [i for i in wanted if i not in by_id]
        if missing:
            ap.error(f"{len(missing)} ids not in manifest, e.g. {missing[:3]}")
        todo_rows = [by_id[i] for i in wanted]
    else:
        todo_rows = list(rows)
    if args.limit:
        todo_rows = todo_rows[: args.limit]

    already = done_ids(args.out_dir)
    todo = [r for r in todo_rows if r["id"] not in already]
    print(f"{len(todo_rows)} requested, {len(already)} already done, {len(todo)} to run", flush=True)
    if args.dry_run or not todo:
        return

    prompt = load_prompt(args.prompt_file, args.prompt_name)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set (looked in .env)")

    (args.out_dir / "run_meta.json").write_text(json.dumps({
        "argv": sys.argv, "args": {k: str(v) for k, v in vars(args).items()},
        "prompt_sha1": hashlib.sha1(prompt.encode()).hexdigest(),
        "prompt_chars": len(prompt), "n_requested": len(todo_rows), "n_todo": len(todo),
        "started": time.time(),
    }, indent=1))

    monitor = Monitor(args.strict, EXPECTED_CACHED_TOKENS)
    handles = {s: open(args.out_dir / f"shard-{s:02d}.jsonl", "a") for s in range(args.shards)}
    write_lock = threading.Lock()

    def emit(rec: dict) -> None:
        with write_lock:
            fh = handles[rec["shard"]]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

    seen_shards: set[int] = set()
    warm: list[dict] = []
    rest: list[dict] = []
    for r in todo:
        s = shard_of(r["id"], args.shards)
        (warm if s not in seen_shards else rest).append(r)
        seen_shards.add(s)
    print(f"warm-up: {len(warm)} request(s), one per shard", flush=True)
    for r in warm:
        rec = label_one(r, prompt, args, api_key, monitor, warmup=True, emit=emit)
        print(f"  warm {rec['id']:18s} in={rec.get('input_tokens')} "
              f"cached={rec.get('cached_tokens')} cw={rec.get('cache_write_tokens')} "
              f"{rec['latency_s']}s", flush=True)
        if monitor.should_abort():
            break
    reason = monitor.should_abort()
    if reason:
        print(f"\nABORT during warm-up: {reason}", flush=True)
        rest = []
    time.sleep(2.0)

    t0 = time.time()
    done = 0
    aborted = False
    it = iter(rest)
    with ThreadPoolExecutor(args.workers) as ex:

        futs: dict = {}

        def submit_next() -> bool:
            try:
                r = next(it)
            except StopIteration:
                return False
            futs[ex.submit(label_one, r, prompt, args, api_key, monitor, False, emit)] = r
            return True

        for _ in range(args.workers * 2):
            if not submit_next():
                break

        while futs and not aborted:
            done_now, _ = wait(list(futs), return_when=FIRST_COMPLETED)
            for fut in done_now:
                row = futs.pop(fut)
                try:
                    fut.result()
                except Exception as e:
                    emit({"id": row["id"], "source": row["source"],
                          "shard": shard_of(row["id"], args.shards),
                          "frames_dir": row["frames_dir"], "warmup": False,
                          "error": f"worker: {type(e).__name__}: {e}",
                          "raw": None, "parsed": None, "parse_ok": False})
                done += 1
                if done % 10 == 0 or done == len(rest):
                    rate = done / max(time.time() - t0, 1e-6) * 60
                    print(f"  {done}/{len(rest)}  hits={monitor.hits} miss={monitor.misses} "
                          f"writes={monitor.writes}  {rate:.0f}/min", flush=True)
                reason = monitor.should_abort()
                if reason:
                    print(f"\nABORT: {reason}", flush=True)
                    aborted = True
                    break
                submit_next()
        if aborted:
            for f in futs:
                f.cancel()

    for fh in handles.values():
        fh.close()

    recs = [json.loads(l) for f in sorted(args.out_dir.glob("shard-*.jsonl"))
            for l in f.read_text().splitlines() if l.strip()]
    ok = [r for r in recs if r.get("parse_ok")]
    print(f"\ndone: {len(recs)} records, {len(ok)} parsed, "
          f"cache hits {monitor.hits}/{monitor.post_warmup} post-warm-up, "
          f"cache writes {monitor.writes}", flush=True)
    if monitor.should_abort():
        sys.exit(2)

if __name__ == "__main__":
    main()
