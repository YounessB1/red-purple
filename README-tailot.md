# Running a tailored agent prompt

This explains how to take a CTF-agent prompt and run it against an XBOW
benchmark, N times, to measure how reliably it solves that benchmark.

## 1. Pull the xbow benchmarks

The benchmarks themselves live in a separate repo, wired in as a git
submodule at `xbow/`. If you cloned without `--recurse-submodules`, pull it
now:

```bash
git submodule update --init --recursive
```

That gives you `xbow/benchmarks/<BENCHMARK_ID>/` for each challenge (its
Dockerfile/docker-compose.yml, app source, and Makefile) plus
`xbow/instructions/` (curated solution writeups, used elsewhere in the
repo, not needed for a normal run). You don't need to `make build` or
otherwise set up any individual benchmark by hand — `source/benchmark.py`
does that automatically (`make build` + `docker compose up -d` under the
hood) the moment you launch an eval; see step 4.

A couple of the older benchmark images were only ever published for
`linux/amd64`. On Apple Silicon this would otherwise fail with "found but
does not provide the specified platform" — the eval scripts already set
`DOCKER_DEFAULT_PLATFORM=linux/amd64` for you, so you don't need to; it's
just worth knowing if you ever invoke `make`/`docker compose` directly
inside a benchmark folder yourself.

## 2. One-time setup

Create a `.env` at the repo root with the LLM provider key:

```
POLITO_API_KEY=<your key>
```

Start the agent-server container — a Kali-based Docker image with the
`opencode` CLI and the usual CTF toolset (curl, sqlmap, nmap, hydra,
gobuster, flask-unsign, etc.) preinstalled. It stays running in the
background; you only need to do this once per session (it survives across
many separate experiment runs):

```bash
docker compose -f source/agent/docker-compose.yml up -d --build
```

## 3. The agent you're editing

The live, editable agent config lives at **`source/tailor/`**:

```
source/tailor/
├── prompt.md          # the prompt itself — edit this
├── opencode.json       # model/provider config
├── AGENTS.md
└── .opencode/agents/ctf-agent.md   # model id, step budget, permissions
```

To try a new prompt, just edit `source/tailor/prompt.md` and save. There's
nothing else to rebuild or restart — the eval script reads this directory
fresh every time it's invoked.

## 4. Run it

```bash
cd /Users/younessbouchari/red-purple
python3 -m source.tailor_replicate_eval --benchmark XBEN-022-24 --try my_experiment --runs 10
```

- `--benchmark` — which XBOW benchmark to target (e.g. `XBEN-022-24`). The
  full list of available ids is under `xbow/benchmarks/`.
- `--try NAME` — names the output folder and snapshots the exact prompt
  you ran (see below), so you keep a permanent, byte-exact record even
  after you edit `source/tailor/prompt.md` again for your next attempt.
- `--runs` — how many replicate runs (default 5). Replicates run
  concurrently by default; add `--sequential` to run them one at a time
  instead (slower, but removes any resource-contention between runs as a
  confound).

A full run takes a few minutes per replicate and uses real LLM calls
(billed against `POLITO_API_KEY`).

## 5. Where the results land

```
experiments/tailor/<BENCHMARK_ID>/<NAME>/
├── agent/                     # snapshot of the exact prompt.md/opencode.json/etc. used
├── run1/<BENCHMARK_ID>/
│   ├── metadata.json          # success, stop_reason, cost, tokens, ...
│   └── context_window.json    # the full step-by-step trajectory (thoughts + commands + outputs)
├── run2/...
└── runs_summary.json          # pass/fail + cost per run, plus the mean pass rate
```

## 6. Keeping results around

`experiments/tailor/` is the scratch area for prompts you're actively
iterating on. Once you're happy with a result and want to keep it as a
permanent comparison point, move or copy the folder into
`experiments/tests/<BENCHMARK_ID>/<NAME>/` — that's the archive other
experiments get compared against.
