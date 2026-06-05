#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Longform Novel Drafting Pipeline

A compact, public-GitHub-friendly pipeline for turning a source manuscript,
outline, or notes that you have rights to use into:
- extracted text chunks
- rolling summaries
- a global memory file
- chapter plans
- original draft chapters
- a combined manuscript file

Important: use this with your own writing, public-domain texts, or material you
have permission to process. Do not upload private source texts, API keys, or
copyrighted source material into a public repository.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import requests

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(items, **_: object):
        return items


DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
DEFAULT_ENDPOINT_TEMPLATE = os.environ.get(
    "GEMINI_ENDPOINT_TEMPLATE",
    "https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent?key={api_key}",
)


@dataclass
class Paths:
    workdir: Path

    @property
    def data_dir(self) -> Path:
        return self.workdir / "data"

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "chunks"

    @property
    def source_summaries_dir(self) -> Path:
        return self.data_dir / "source_summaries"

    @property
    def memory_dir(self) -> Path:
        return self.workdir / "memory"

    @property
    def output_dir(self) -> Path:
        return self.workdir / "output"

    @property
    def plans_dir(self) -> Path:
        return self.output_dir / "plans"

    @property
    def chapters_dir(self) -> Path:
        return self.output_dir / "chapters"

    @property
    def chapter_summaries_dir(self) -> Path:
        return self.output_dir / "summaries"

    @property
    def full_text_path(self) -> Path:
        return self.data_dir / "full_text.txt"

    @property
    def global_memory_path(self) -> Path:
        return self.memory_dir / "global_memory.md"

    @property
    def state_path(self) -> Path:
        return self.output_dir / "pipeline_state.json"

    @property
    def combined_path(self) -> Path:
        return self.output_dir / "combined_manuscript.md"

    def ensure(self) -> None:
        for path in [
            self.workdir,
            self.data_dir,
            self.chunks_dir,
            self.source_summaries_dir,
            self.memory_dir,
            self.output_dir,
            self.plans_dir,
            self.chapters_dir,
            self.chapter_summaries_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


class CostTracker:
    """Approximate local guardrail only. Real billing is determined by provider billing."""

    def __init__(self, limit_usd: float):
        self.limit_usd = limit_usd
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def record(self, usage: dict) -> None:
        self.prompt_tokens += int(usage.get("promptTokenCount", 0) or 0)
        self.output_tokens += int(usage.get("candidatesTokenCount", 0) or 0)
        self.calls += 1

    def rough_cost(self) -> float:
        return self.prompt_tokens / 1_000_000 * 4.0 + self.output_tokens / 1_000_000 * 18.0

    def check(self) -> None:
        if self.rough_cost() > self.limit_usd:
            raise RuntimeError(
                f"Local soft budget reached: approximately ${self.rough_cost():.3f} / ${self.limit_usd:.2f}. "
                "Increase --budget only after checking your cloud billing settings."
            )

    def status(self) -> str:
        return (
            f"calls={self.calls}, input={self.prompt_tokens / 1000:.1f}K, "
            f"output={self.output_tokens / 1000:.1f}K, rough_cost=${self.rough_cost():.4f}/${self.limit_usd:.2f}"
        )


class GeminiClient:
    def __init__(self, model: str, budget_usd: float, endpoint_template: str = DEFAULT_ENDPOINT_TEMPLATE):
        self.model = model
        self.endpoint_template = endpoint_template
        self.cost = CostTracker(budget_usd)

    @staticmethod
    def api_key() -> str:
        key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY. Set it in your shell or in a local .env file that is not committed."
            )
        return key

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.5,
        max_output_tokens: int = 8192,
        retries: int = 4,
        timeout: int = 300,
    ) -> str:
        url = self.endpoint_template.format(model=self.model, api_key=self.api_key())
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": max_output_tokens,
            },
        }

        last_error = "unknown error"
        for attempt in range(retries):
            try:
                response = requests.post(url, json=payload, timeout=timeout)
                if response.status_code == 200:
                    data = response.json()
                    self.cost.record(data.get("usageMetadata", {}))
                    self.cost.check()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        raise RuntimeError(f"No candidates returned: {json.dumps(data, ensure_ascii=False)[:800]}")
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if not parts or "text" not in parts[0]:
                        raise RuntimeError(f"No text returned: {json.dumps(data, ensure_ascii=False)[:800]}")
                    return parts[0]["text"].strip()

                last_error = f"HTTP {response.status_code}: {response.text[:800]}"
            except Exception as exc:  # noqa: BLE001
                last_error = repr(exc)

            wait = 3 + attempt * 6 + random.random() * 2
            print(f"[WARN] model call failed, retrying in {wait:.1f}s: {last_error[:200]}")
            time.sleep(wait)

        raise RuntimeError(f"Model call failed after retries: {last_error}")


def load_state(paths: Paths) -> dict:
    if not paths.state_path.exists():
        return {}
    try:
        return json.loads(paths.state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(paths: Paths, key: str, value: object) -> None:
    paths.ensure()
    state = load_state(paths)
    state[key] = value
    tmp = paths.state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(paths.state_path)


def read_text(path: Path, tail_chars: Optional[int] = None) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[-tail_chars:] if tail_chars else text


def extract_source(paths: Paths, source: Path) -> None:
    paths.ensure()
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    if source.suffix.lower() == ".pdf":
        if fitz is None:
            raise RuntimeError("PyMuPDF is required for PDF extraction. Install with: pip install pymupdf")
        doc = fitz.open(source)
        pages = []
        for i in tqdm(range(len(doc)), desc="Extracting PDF"):
            text = doc[i].get_text("text")
            if text.strip():
                pages.append(text)
        full_text = "\n\n".join(pages)
    else:
        full_text = source.read_text(encoding="utf-8", errors="ignore")

    paths.full_text_path.write_text(full_text, encoding="utf-8")
    print(f"Saved extracted text to {paths.full_text_path} ({len(full_text):,} characters).")


def split_source(paths: Paths, chunk_size: int = 18000) -> None:
    paths.ensure()
    if not paths.full_text_path.exists():
        raise FileNotFoundError("Missing data/full_text.txt. Run extract first.")

    for old in paths.chunks_dir.glob("*.txt"):
        old.unlink()

    text = read_text(paths.full_text_path)
    pattern = re.compile(r"(?m)(^\s*第\s*[0-9零一二三四五六七八九十百千万两]+\s*章[^\n\r]{0,80})")
    matches = list(pattern.finditer(text))

    if len(matches) >= 10:
        for idx, match in enumerate(tqdm(matches, desc="Writing chapters"), 1):
            start = match.start()
            end = matches[idx].start() if idx < len(matches) else len(text)
            title = re.sub(r"\s+", " ", match.group(1).strip())
            safe_title = re.sub(r'[\\/:*?"<>|]', "_", title)[:70]
            out = paths.chunks_dir / f"{idx:04d}_{safe_title}.txt"
            out.write_text(text[start:end].strip(), encoding="utf-8")
    else:
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        for idx, chunk in enumerate(tqdm(chunks, desc="Writing chunks"), 1):
            (paths.chunks_dir / f"chunk_{idx:04d}.txt").write_text(chunk, encoding="utf-8")

    count = len(list(paths.chunks_dir.glob("*.txt")))
    print(f"Saved {count} chunks to {paths.chunks_dir}.")


def chunk_files(paths: Paths) -> list[Path]:
    files = sorted(paths.chunks_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError("No chunk files found. Run split first.")
    return files


def summarize_source_tail(paths: Paths, client: GeminiClient, n: int = 5) -> None:
    paths.ensure()
    files = chunk_files(paths)[-n:]
    for file in files:
        out = paths.source_summaries_dir / f"{file.stem}.summary.md"
        if out.exists():
            print(f"[SKIP] {out.name}")
            continue
        text = read_text(file, tail_chars=30000)
        prompt = f"""
You are preparing continuity notes for an original long-form fiction project.
Summarize the following source chunk without copying wording.

Only extract high-level continuity information that the user is allowed to reuse.
Do not imitate a living author's distinctive style.

Chunk name: {file.name}

SOURCE CHUNK:
{text}

Return:
## Plot Progress
## Character State
## Factions and Interests
## Open Threads
## Continuity Rules
"""
        print(f"Summarizing source chunk: {file.name}")
        result = client.generate(prompt, temperature=0.25, max_output_tokens=4096)
        out.write_text(result, encoding="utf-8")
    print(client.cost.status())


def make_global_memory(paths: Paths, client: GeminiClient, samples: int = 30) -> None:
    paths.ensure()
    if paths.global_memory_path.exists():
        print(f"Global memory already exists: {paths.global_memory_path}")
        return

    files = chunk_files(paths)
    step = max(1, len(files) // max(1, min(samples, len(files))))
    chosen = files[::step][:samples]
    pieces = []
    for file in chosen:
        pieces.append(f"### {file.name}\n{read_text(file)[:7000]}")

    prompt = f"""
You are building a global memory file for an original long-form fiction project.
Use the excerpts only as high-level reference notes. Do not copy source wording,
do not imitate a living author's distinctive style, and do not reproduce protected text.

EXCERPTS:
{"\n\n".join(pieces)}

Return a compact memory file with:
## World Premise
## Power / Rule System
## Protagonist Profile
## Key Characters
## Major Factions
## Core Conflicts
## Tone Guidelines
## Publishing Safety Notes
## Continuity Rules
"""
    print("Generating global memory...")
    result = client.generate(prompt, temperature=0.25, max_output_tokens=8192, timeout=600)
    paths.global_memory_path.write_text(result, encoding="utf-8")
    print(f"Saved global memory to {paths.global_memory_path}")
    print(client.cost.status())


def summary_files(paths: Paths, up_to_chapter: int) -> list[Path]:
    files = sorted(paths.source_summaries_dir.glob("*.summary.md"))
    for i in range(1, up_to_chapter):
        path = paths.chapter_summaries_dir / f"chapter_{i:03d}.summary.md"
        if path.exists():
            files.append(path)
    return files


def recent_summaries(paths: Paths, up_to_chapter: int, n: int = 5) -> str:
    files = summary_files(paths, up_to_chapter)[-n:]
    return "\n\n---\n\n".join(f"# {f.stem}\n\n{read_text(f)}" for f in files)


def recent_full_texts(paths: Paths, up_to_chapter: int, n: int = 2) -> str:
    files = chunk_files(paths)
    for i in range(1, up_to_chapter):
        path = paths.chapters_dir / f"chapter_{i:03d}.md"
        if path.exists():
            files.append(path)
    recent = files[-n:]
    return "\n\n".join(f"### {f.name}\n{read_text(f, tail_chars=20000)}" for f in recent)


def plan_chapter(paths: Paths, client: GeminiClient, chapter_num: int, summary_n: int, fulltext_n: int) -> Path:
    paths.ensure()
    out = paths.plans_dir / f"plan_{chapter_num:03d}.md"
    if out.exists():
        print(f"[SKIP] plan exists: {out.name}")
        return out

    memory = read_text(paths.global_memory_path) if paths.global_memory_path.exists() else ""
    prompt = f"""
You are a long-form fiction planning assistant.
Create a detailed plan for original chapter {chapter_num}.

Rules:
- Write original material only.
- Do not copy source wording or imitate a living author's distinctive style.
- Preserve high-level continuity, character goals, faction pressure, and unresolved threads.
- Prefer strategy, tradeoffs, dialogue tension, consequences, and a strong ending hook.

GLOBAL MEMORY:
{memory}

RECENT SUMMARIES:
{recent_summaries(paths, chapter_num, summary_n)}

RECENT TEXT EXCERPTS:
{recent_full_texts(paths, chapter_num, fulltext_n)}

Return:
## Chapter Title
## Core Objective
## Opening Continuity
## Main Conflict
## Character Decision Chain
## Scene 1
## Scene 2
## Scene 3
## Ending Hook
## Risk Checks
"""
    print(f"Planning chapter {chapter_num}...")
    result = client.generate(prompt, temperature=0.55, max_output_tokens=8192, timeout=600)
    out.write_text(result, encoding="utf-8")
    print(client.cost.status())
    return out


def draft_chapter(paths: Paths, client: GeminiClient, chapter_num: int, words: int, summary_n: int, fulltext_n: int) -> Path:
    paths.ensure()
    out = paths.chapters_dir / f"chapter_{chapter_num:03d}.md"
    if out.exists():
        print(f"[SKIP] draft exists: {out.name}")
        return out

    plan_path = paths.plans_dir / f"plan_{chapter_num:03d}.md"
    if not plan_path.exists():
        raise FileNotFoundError("Missing chapter plan. Run plan first.")

    memory = read_text(paths.global_memory_path) if paths.global_memory_path.exists() else ""
    prompt = f"""
You are a long-form fiction drafting assistant.
Write original chapter {chapter_num} based on the plan.

Hard rules:
1. Original text only; do not copy source wording.
2. Do not imitate a living author's distinctive style.
3. Keep continuity with the provided high-level memory and recent summaries.
4. Use concrete action, dialogue, setting details, and psychological tradeoffs.
5. Avoid gratuitous cruelty, sexual content, or extremist praise.
6. Target length: about {words} Chinese characters or equivalent prose length.
7. Output only the chapter text.

GLOBAL MEMORY:
{memory}

RECENT SUMMARIES:
{recent_summaries(paths, chapter_num, summary_n)}

RECENT TEXT EXCERPTS:
{recent_full_texts(paths, chapter_num, fulltext_n)}

CHAPTER PLAN:
{read_text(plan_path)}
"""
    print(f"Drafting chapter {chapter_num}...")
    result = client.generate(prompt, temperature=0.85, max_output_tokens=12000, timeout=600)
    out.write_text(result, encoding="utf-8")
    print(f"Saved chapter to {out} ({len(result):,} characters).")
    print(client.cost.status())
    return out


def summarize_chapter(paths: Paths, client: GeminiClient, chapter_num: int) -> Path:
    paths.ensure()
    chapter_path = paths.chapters_dir / f"chapter_{chapter_num:03d}.md"
    if not chapter_path.exists():
        raise FileNotFoundError("Missing chapter draft. Run draft first.")
    out = paths.chapter_summaries_dir / f"chapter_{chapter_num:03d}.summary.md"
    if out.exists():
        print(f"[SKIP] chapter summary exists: {out.name}")
        return out

    prompt = f"""
Summarize this newly drafted chapter for rolling continuity memory.
Do not copy long passages. Extract only the information needed for future chapters.

CHAPTER {chapter_num}:
{read_text(chapter_path, tail_chars=30000)}

Return:
## Chapter Summary
## Character Changes
## Factions and Interests
## Open Threads
## Next Chapter Must Continue
"""
    print(f"Summarizing chapter {chapter_num}...")
    result = client.generate(prompt, temperature=0.2, max_output_tokens=4096)
    out.write_text(result, encoding="utf-8")
    print(client.cost.status())
    return out


def combine(paths: Paths) -> None:
    paths.ensure()
    chapters = sorted(paths.chapters_dir.glob("chapter_*.md"))
    parts = []
    for chapter in chapters:
        parts.append(f"# {chapter.stem}\n\n{read_text(chapter)}")
    paths.combined_path.write_text("\n\n---\n\n".join(parts), encoding="utf-8")
    print(f"Combined {len(chapters)} chapters into {paths.combined_path}.")


def run_pipeline(args: argparse.Namespace) -> None:
    paths = Paths(Path(args.workdir).resolve())
    client = GeminiClient(args.model, args.budget)
    paths.ensure()

    if args.source:
        extract_source(paths, Path(args.source).resolve())
    if not list(paths.chunks_dir.glob("*.txt")):
        split_source(paths, chunk_size=args.chunk_size)
    summarize_source_tail(paths, client, n=args.recent_source)
    make_global_memory(paths, client, samples=args.samples)

    state = load_state(paths)
    completed = int(state.get("chapters_completed", 0) or 0)
    for chapter_num in range(completed + 1, args.chapters + 1):
        plan_chapter(paths, client, chapter_num, args.summary_n, args.fulltext_n)
        draft_chapter(paths, client, chapter_num, args.words, args.summary_n, args.fulltext_n)
        summarize_chapter(paths, client, chapter_num)
        save_state(paths, "chapters_completed", chapter_num)
    combine(paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Longform novel planning and drafting pipeline")
    parser.add_argument("cmd", choices=["extract", "split", "summarize", "memory", "plan", "draft", "chapter", "combine", "pipeline"])
    parser.add_argument("--workdir", default="workspace", help="Working directory for data, memory, and output")
    parser.add_argument("--source", help="Source PDF or TXT file that you have rights to process")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--budget", type=float, default=8.0, help="Approximate local soft budget in USD")
    parser.add_argument("--chunk-size", type=int, default=18000)
    parser.add_argument("--recent-source", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--chapter", type=int, default=1)
    parser.add_argument("--chapters", type=int, default=3)
    parser.add_argument("--words", type=int, default=3000)
    parser.add_argument("--summary-n", type=int, default=5)
    parser.add_argument("--fulltext-n", type=int, default=2)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = Paths(Path(args.workdir).resolve())
    client = GeminiClient(args.model, args.budget)
    paths.ensure()

    if args.cmd == "extract":
        if not args.source:
            raise RuntimeError("extract requires --source")
        extract_source(paths, Path(args.source).resolve())
    elif args.cmd == "split":
        split_source(paths, chunk_size=args.chunk_size)
    elif args.cmd == "summarize":
        summarize_source_tail(paths, client, n=args.recent_source)
    elif args.cmd == "memory":
        make_global_memory(paths, client, samples=args.samples)
    elif args.cmd == "plan":
        plan_chapter(paths, client, args.chapter, args.summary_n, args.fulltext_n)
    elif args.cmd == "draft":
        draft_chapter(paths, client, args.chapter, args.words, args.summary_n, args.fulltext_n)
    elif args.cmd == "chapter":
        plan_chapter(paths, client, args.chapter, args.summary_n, args.fulltext_n)
        draft_chapter(paths, client, args.chapter, args.words, args.summary_n, args.fulltext_n)
        summarize_chapter(paths, client, args.chapter)
    elif args.cmd == "combine":
        combine(paths)
    elif args.cmd == "pipeline":
        run_pipeline(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Existing output files are preserved.")
