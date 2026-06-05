# Longform Novel Drafting Toolkit

A compact, public-GitHub-friendly Python pipeline for planning and drafting original long-form fiction with a Gemini/Vertex-style model endpoint.

It can:

1. extract text from a PDF or TXT file,
2. split the source into chapters or chunks,
3. create rolling summaries,
4. build a global memory file,
5. generate chapter plans,
6. draft original chapters,
7. summarize generated chapters for continuity,
8. combine generated chapters into one manuscript file.

## Important copyright note

Use this project with material you wrote yourself, public-domain material, or source material you have permission to process. Do not commit copyrighted novels, private PDFs, generated fanfiction based on protected works, API keys, service-account JSON files, or private notes to a public repository.

This repository is meant to demonstrate the pipeline code, not to distribute source novels or generated derivative text.

## Install

```bash
pip install -r requirements.txt
```

## Configure

Set your API key locally. Do not commit it.

```bash
export GEMINI_API_KEY="your_api_key_here"
```

On Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

You can optionally override the model:

```bash
export GEMINI_MODEL="gemini-3.1-pro-preview"
```

## Example usage

Extract a source file you have rights to process:

```bash
python longform_novel_pipeline.py extract --source ./my_notes_or_manuscript.pdf --workdir ./workspace
```

Split it:

```bash
python longform_novel_pipeline.py split --workdir ./workspace
```

Summarize recent source chunks:

```bash
python longform_novel_pipeline.py summarize --workdir ./workspace --recent-source 5
```

Build global memory:

```bash
python longform_novel_pipeline.py memory --workdir ./workspace --samples 30
```

Generate one chapter:

```bash
python longform_novel_pipeline.py chapter --workdir ./workspace --chapter 1 --words 3000
```

Run a small pipeline:

```bash
python longform_novel_pipeline.py pipeline --source ./my_notes_or_manuscript.pdf --workdir ./workspace --chapters 3 --words 3000
```

## Output structure

```text
workspace/
  data/
    full_text.txt
    chunks/
    source_summaries/
  memory/
    global_memory.md
  output/
    plans/
    chapters/
    summaries/
    combined_manuscript.md
    pipeline_state.json
```

## Why this repo is generic

The original local scripts were built for one personal writing experiment. This public version removes:

- local Windows paths,
- service-account paths,
- project IDs,
- bucket names,
- private source titles,
- generated private chapters,
- hardcoded credentials.

That makes the code safer and more useful as a general writing pipeline.
