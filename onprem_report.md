# Using the On-Prem Model with OpenCode

How this repo drives PoliTO's on-prem LLM server through the OpenCode agent backend, and how to run it yourself.

## 1. What the on-prem provider is

PoliTO (Politecnico di Torino) runs its own OpenAI-compatible inference server at `https://llm.polito.it/v1`, on a single GPU. It is declared as an OpenCode provider config, not hard-coded into the harness — the same `OpenCodeBackend` class that talks to this server can talk to OpenRouter or any other OpenAI-compatible endpoint by pointing at a different config file.

The provider config lives at [`experiments/opencode/polito.json`](../experiments/opencode/polito.json) (committed, no secrets):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "autoupdate": false,
  "provider": {
    "polito": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "PoliTO on-prem LLM",
      "options": {
        "baseURL": "https://llm.polito.it/v1",
        "apiKey": "{env:POLITO_API_KEY}"
      },
      "models": {
        "qwen3-coder-next": { "name": "Qwen3 Coder Next (FP8)", "limit": { "context": 262144, "output": 8192 } },
        "qwen3-235b":       { "name": "Qwen3 235B Instruct-2507 (AWQ)", "limit": { "context": 262144, "output": 8192 } },
        "mistral-small-4":  { "name": "Mistral Small 4 (text)", "limit": { "context": 327680, "output": 8192 } },
        "gemma-4-31b":      { "name": "Gemma 4 31B", "limit": { "context": 262144, "output": 8192 } },
        "gpt-oss-120b":     { "name": "GPT-OSS 120B", "limit": { "context": 131072, "output": 8192 }, "options": { "reasoning_effort": "low" } },
        "gpt-oss-20b":      { "name": "GPT-OSS 20B", "limit": { "context": 131072, "output": 8192 }, "options": { "reasoning_effort": "low" } }
      }
    }
  }
}
```

A model is addressed to OpenCode as `<provider>/<model-id>` — for this provider that's `polito/qwen3-235b`, `polito/gpt-oss-120b`, etc. Because the box has one GPU, only one model is loaded at a time — the harness treats these six as a roster to run **sequentially**, never concurrently.

## 2. Credentials

OpenCode resolves `{env:POLITO_API_KEY}` in the config against the container's environment. The key itself lives in a **gitignored** env file, `experiments/opencode/polito.env`, in simple `KEY=VALUE` form:

```bash
POLITO_API_KEY=your-key-here
```

This file does not exist in the repo yet — you have to create it. Get the key from whoever administers `llm.polito.it` (it's not a public/self-serve endpoint).

## 3. How a call actually happens

Nothing about the on-prem case is special-cased — it's the generic OpenCode path in [`experiments/agents/opencode.py`](../experiments/agents/opencode.py):

1. **`--env-file experiments/opencode/polito.env`** is parsed (`load_env_file`) and merged into the agent container's env (`docker run -e POLITO_API_KEY=...`).
2. **`--opencode-config experiments/opencode/polito.json`** is bind-mounted read-only into the agent container at `/seed/opencode.json`, with `OPENCODE_CONFIG=/seed/opencode.json` set so OpenCode loads it.
3. Inside the container (built from [`experiments/agents/docker/opencode.Dockerfile`](../experiments/agents/docker/opencode.Dockerfile) — `node:20-bookworm-slim` + the `opencode-ai` npm package, non-root `agent` user), the harness runs:

   ```bash
   opencode run "<challenge prompt>" \
     --agent ctf-agent \
     -m polito/gpt-oss-120b \
     --format json \
     --dangerously-skip-permissions \
     --thinking
   ```

4. `--agent ctf-agent` points at [`experiments/agents/opencode_agents/ctf-agent.md`](../experiments/agents/opencode_agents/ctf-agent.md), copied into the sandbox's `/work/.opencode/agents/` before launch. It fixes `temperature: 0.0`, `maxSteps: 200`, tool permissions (`bash/read/write/edit/glob/grep: allow`, `webfetch/websearch/task: deny`), and the CTF system prompt — so the model only ever gets there over HTTP against the target container, no web access.
5. The agent container itself has no docker socket, no repo mount, no results mount — only the private per-trial network (challenge reachable at `http://target:5000`) and a `/work` scratch dir. It cannot reach the flag except by solving the challenge.
6. `--format json` streams line-delimited events, parsed back by `summarize_opencode_stream` into a transcript plus cost/token/step metrics — even from a truncated stream if the trial times out.

## 4. Running it

### Quick sanity check (no cost)

```bash
python experiments/run.py \
  --backend opencode --model polito/gpt-oss-120b \
  --opencode-config experiments/opencode/polito.json \
  --env-file experiments/opencode/polito.env \
  --challenge sqli-simple --dry-run
```

### A single live run against one challenge

```bash
python experiments/run.py \
  --backend opencode --model polito/gpt-oss-120b \
  --opencode-config experiments/opencode/polito.json \
  --env-file experiments/opencode/polito.env \
  --challenge sqli-simple --trials 1
```

### The full baseline/deception rounds

[`experiments/scripts/polito_baseline.sh`](../experiments/scripts/polito_baseline.sh) and [`polito_deceptions.sh`](../experiments/scripts/polito_deceptions.sh) wrap `experiments/suite.py` with the fixed six-model roster, dry-run by default:

```bash
./experiments/scripts/polito_baseline.sh                          # plan only
DRYRUN=0 ./experiments/scripts/polito_baseline.sh                 # run all 6 models, baseline
DRYRUN=0 MODELS="polito/gpt-oss-120b" ./experiments/scripts/polito_baseline.sh   # just one model
DRYRUN=0 RESUME=1 ./experiments/scripts/polito_deceptions.sh      # finish an interrupted round
```

Models in `MODELS` run one at a time (one `suite_<ts>/` each), matching the single-GPU constraint. `CONCURRENCY` (default 8) controls parallel *trials within* one model's run, not across models.

## 5. Adding a different on-prem endpoint

Nothing above is PoliTO-specific beyond the JSON file. To point at another OpenAI-compatible server:

```bash
python experiments/run.py --backend opencode --print-config-template
```

prints a template you can fill in (`baseURL`, `apiKey` env var name, model IDs + context/output limits), save as e.g. `experiments/opencode/myendpoint.json`, pair with an `experiments/opencode/myendpoint.env`, and pass via the same `--opencode-config` / `--env-file` flags.

## 6. Notes / gotchas

- **`--allowed-tools` is ignored** by the OpenCode backend — tool access comes entirely from `ctf-agent.md`'s `permission:` block, not the harness CLI flag (a warning is logged once if you pass it).
- **`maxSteps: 200`** in the agent file is the ground truth for the step cap; if OpenCode stops a trial after using exactly that many steps, `summarize()` flags `max_steps_hit` so the trial is recorded as `stop_reason="max_steps"` rather than a genuine completion.
- **Cost is $0 per token** (on-prem), but time is real: `TIMEOUT` defaults to 900s per trial in the launch scripts specifically because trials queue behind each other on the one GPU under concurrency.
- Only the maintained deception instance set (`deceptions/instances/generated/manifest.jsonl`) is used by `polito_deceptions.sh` — orphaned `pixel_*`/`hybrid_*` instances are excluded automatically.
