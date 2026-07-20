"""Real semantic-embedding similarity between GEPA candidates, via OpenRouter.

Computed once at `visualize.py` generation time (not in the browser) so the
API key never reaches the static HTML. Results are cached per-experiment,
keyed by content hash — one entry per *file* (not per candidate), so an
unchanged skill file carried across many mutations is only ever embedded once.

Requires OPENROUTER_API_KEY. If it's missing, or the request fails for any
reason (no network, bad key, model unavailable, ...), similarity is marked
unavailable with a human-readable reason instead of failing the whole run —
the rest of the viewer still generates normally.
"""
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

EMBEDDING_MODEL = os.environ.get("RESULT_ANALYSIS_EMBEDDING_MODEL", "openai/text-embedding-3-small")
OPENROUTER_URL = "https://openrouter.ai/api/v1/embeddings"

_EXCLUDE_FILES = ["ctf-agent.md", "opencode.json"]  # static boilerplate, identical across every candidate
_CHUNK_CHARS = 24_000  # ~8k tokens, the model's real per-input hard limit — only a
                       # defensive fallback here, since a single file (prompt.md,
                       # AGENTS.md, one SKILL.md) is always well under this in practice
_BATCH_SIZE = 20
_TIMEOUT = 30.0
_SEED_DIR = Path(__file__).resolve().parents[3] / "source" / "seed"


def _load_seed_files() -> dict:
    """iteration_000's 'parent' snapshot (the seed) isn't always captured on disk,
    so fall back to the actual seed source GEPA starts from."""
    if not _SEED_DIR.exists():
        return {}
    files = {}
    for f in sorted(_SEED_DIR.rglob("*")):
        if f.is_file():
            try:
                files[str(f.relative_to(_SEED_DIR))] = f.read_text(encoding="utf-8")
            except Exception:
                pass
    return files


def build_evolution_nodes(iterations: list[dict]) -> list[dict]:
    """Seed + accepted candidates only — mirrors buildEvolutionTree() in evolution.js
    so the Similarity tab compares exactly the same candidate set as Tree/Validation."""
    nodes: dict[int, dict] = {}
    seed_iter = iterations[0] if iterations else None
    seed_files = (seed_iter or {}).get("parent_files") or {}
    if not seed_files:
        seed_files = _load_seed_files()
    nodes[0] = {"idx": 0, "iter": 0, "files": seed_files}
    for it in iterations:
        if it.get("status") != "accepted":
            continue
        child_files = it.get("child_files") or {}
        if not child_files:
            continue
        new_idx = it.get("child_candidate_idx")
        if new_idx is None or new_idx == 0:
            continue
        nodes[new_idx] = {"idx": new_idx, "iter": it["id"], "files": child_files}
    return sorted(nodes.values(), key=lambda n: n["idx"])


def _chunk_text(text: str, size: int = _CHUNK_CHARS) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def _file_chunks(files: dict) -> list[str]:
    """One chunk of text per included file — file boundaries, not blind byte
    offsets, so every chunk is a single coherent unit (one skill, or AGENTS.md,
    or prompt.md) that's comparable across candidates regardless of how their
    other files' lengths happen to line up. Only sub-splits (defensively — not
    hit by any file in practice) if a single file itself exceeds the model's
    real input limit."""
    out = []
    for path in sorted(files):
        if any(x in path for x in _EXCLUDE_FILES):
            continue
        content = files[path]
        if not content:
            continue
        pieces = _chunk_text(content) if len(content) > _CHUNK_CHARS else [content]
        for piece in pieces:
            out.append(f"# {path}\n{piece}")
    return out


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(path: Path, cache: dict) -> None:
    path.write_text(json.dumps(cache), encoding="utf-8")


def _embed_batch(texts: list[str], api_key: str) -> list[list[float]]:
    body = json.dumps({"model": EMBEDDING_MODEL, "input": texts}).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if "data" not in payload:
        err = payload.get("error", payload)
        raise RuntimeError(f"unexpected response: {err}")
    ordered = sorted(payload["data"], key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in ordered]


def compute_similarity(iterations: list[dict], experiment_dir: Path) -> dict:
    nodes = build_evolution_nodes(iterations)
    if len(nodes) < 2:
        return {"available": False, "reason": "Not enough accepted candidates yet to compare."}

    cache_path = experiment_dir / ".embeddings_cache.json"
    cache = _load_cache(cache_path)

    # Each candidate becomes a list of per-file chunk hashes — file-level, not one
    # pooled vector per candidate — so candidates are compared file-by-file below.
    node_hashes: list[list[str]] = []
    hash_to_text: dict[str, str] = {}
    for n in nodes:
        hashes = []
        for text in _file_chunks(n["files"]):
            h = _hash_text(text)
            hashes.append(h)
            hash_to_text.setdefault(h, text)
        node_hashes.append(hashes)

    all_hashes = sorted(hash_to_text)
    to_fetch = [h for h in all_hashes if h not in cache]

    # Only the API key is needed when there's something new to embed — once every
    # file-chunk here is already in the cache from a prior run, the heatmap should
    # keep working without OPENROUTER_API_KEY set.
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if to_fetch and not api_key:
        return {
            "available": False,
            "reason": "OPENROUTER_API_KEY is not set — set it to enable semantic similarity via embeddings.",
        }

    if to_fetch:
        print(
            f"[visualize] embedding {len(to_fetch)}/{len(all_hashes)} file-chunks "
            f"across {len(nodes)} candidates via {EMBEDDING_MODEL}..."
        )
    try:
        for start in range(0, len(to_fetch), _BATCH_SIZE):
            batch = to_fetch[start:start + _BATCH_SIZE]
            vectors = _embed_batch([hash_to_text[h] for h in batch], api_key)
            for h, vec in zip(batch, vectors):
                cache[h] = vec
        if to_fetch:
            _save_cache(cache_path, cache)
    except urllib.error.HTTPError as e:
        return {"available": False, "reason": f"Embedding request failed: HTTP {e.code} {e.reason}"}
    except urllib.error.URLError as e:
        return {"available": False, "reason": f"Embedding request failed: {e.reason}"}
    except Exception as e:
        return {"available": False, "reason": f"Embedding request failed: {e}"}

    # Mean-center every chunk vector by the corpus-wide mean, over every file-chunk
    # actually used by this run (not the whole possibly-stale cache): every file
    # here — even different skills — shares a dominant "security-agent markdown"
    # direction that embedding models pack into a narrow cone, so raw cosine
    # similarity between any two chunks sits high regardless of how different they
    # actually are. Subtracting the mean exposes how a chunk differs from a
    # *typical* chunk in this population, not absolute closeness in embedding space.
    dim = len(cache[all_hashes[0]])
    mean_vec = [0.0] * dim
    for h in all_hashes:
        v = cache[h]
        for d in range(dim):
            mean_vec[d] += v[d]
    for d in range(dim):
        mean_vec[d] /= len(all_hashes)

    centered = {h: [cache[h][d] - mean_vec[d] for d in range(dim)] for h in all_hashes}
    norms = {h: sum(x * x for x in centered[h]) ** 0.5 for h in all_hashes}

    def chunk_cos(h1: str, h2: str) -> float:
        n1, n2 = norms[h1], norms[h2]
        if n1 == 0 or n2 == 0:
            return 0.0
        v1, v2 = centered[h1], centered[h2]
        return sum(a * b for a, b in zip(v1, v2)) / (n1 * n2)

    # One-to-one matching, not independent per-side max: once a chunk is claimed
    # as the match for something on the other side, it can't also stand in as the
    # match for a different chunk — each file represents at most one
    # correspondence. Greedy: repeatedly take the single highest-similarity pair
    # remaining anywhere in the cross-matrix, commit it, remove both chunks from
    # further consideration, repeat until the smaller side is exhausted. (This is
    # a greedy approximation to the optimal maximum-weight bipartite assignment —
    # a reasonable proxy given how small these per-candidate file counts are, but
    # not mathematically guaranteed identical to the true optimum.) Once the
    # smaller side is fully claimed, any leftover files on the larger side
    # genuinely have no correspondence left — they score 0, not a reused
    # best-available match, and the denominator is the larger file count so they
    # still pull the pair's average down.
    n = len(nodes)
    raw = [[0.0] * n for _ in range(n)]
    pair_sum, pair_count = 0.0, 0
    for i in range(n):
        hashes_i = node_hashes[i]
        for j in range(i + 1, n):
            hashes_j = node_hashes[j]
            pairs = sorted(
                (
                    (chunk_cos(hi, hj), a, b)
                    for a, hi in enumerate(hashes_i)
                    for b, hj in enumerate(hashes_j)
                ),
                key=lambda t: -t[0],
            )
            used_a, used_b = set(), set()
            target = min(len(hashes_i), len(hashes_j))
            matched_sum = 0.0
            for sim_val, a, b in pairs:
                if a in used_a or b in used_b:
                    continue
                used_a.add(a)
                used_b.add(b)
                matched_sum += sim_val
                if len(used_a) == target:
                    break
            sim = matched_sum / max(len(hashes_i), len(hashes_j))
            raw[i][j] = sim
            raw[j][i] = sim
            pair_sum += sim
            pair_count += 1

    # Matching (even exclusive, one-to-one) is still an optimistic statistic on its
    # own — greedily claiming the highest-similarity pairs still skews every score
    # upward (two candidates sharing even a couple of literally-unchanged inherited
    # files already get 1.0 on those), so the raw matrix here comes out all-positive
    # with no natural zero. Re-center the *output* pairwise scores (not the input
    # vectors — this is a second, independent centering step) around their own
    # population mean, so 0 is restored as "a typical pair this run," matching the
    # diverging color scale.
    pair_mean = pair_sum / pair_count if pair_count else 0.0
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(n):
            if i == j:
                continue
            matrix[i][j] = raw[i][j] - pair_mean

    return {
        "available": True,
        "model": EMBEDDING_MODEL,
        "centered": True,
        "chunked": True,
        "items": [{"idx": node["idx"], "iter": node["iter"]} for node in nodes],
        "matrix": matrix,
    }


def _candidate_text_legacy(files: dict) -> str:
    """The original (pre-chunking) whole-candidate text: every included file
    concatenated into one string, truncated at _CHUNK_CHARS — kept around, unchanged,
    so the legacy heatmap reproduces exactly what the very first version of this
    tool computed, truncation warts and all, for side-by-side comparison."""
    parts = []
    for path in sorted(files):
        if any(x in path for x in _EXCLUDE_FILES):
            continue
        content = files[path]
        if content:
            parts.append(f"# {path}\n{content}")
    return "\n\n".join(parts)[:_CHUNK_CHARS]


def compute_similarity_legacy(iterations: list[dict], experiment_dir: Path) -> dict:
    """The original approach, preserved as-is for comparison against the file-level
    one-to-one matching above: one pooled vector per whole candidate, truncated at
    ~8k tokens, with a single population-mean centering step. Shares the same
    on-disk cache file as compute_similarity — the two use disjoint hash spaces
    (whole-candidate text vs. per-file text), so they don't collide or interfere."""
    nodes = build_evolution_nodes(iterations)
    if len(nodes) < 2:
        return {"available": False, "reason": "Not enough accepted candidates yet to compare."}

    cache_path = experiment_dir / ".embeddings_cache.json"
    cache = _load_cache(cache_path)

    texts = [_candidate_text_legacy(n["files"]) for n in nodes]
    hashes = [_hash_text(t) for t in texts]
    to_fetch = [i for i, h in enumerate(hashes) if h not in cache]

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if to_fetch and not api_key:
        return {
            "available": False,
            "reason": "OPENROUTER_API_KEY is not set — set it to enable semantic similarity via embeddings.",
        }

    if to_fetch:
        print(f"[visualize] embedding {len(to_fetch)}/{len(nodes)} candidates (legacy, whole-candidate) via {EMBEDDING_MODEL}...")
    try:
        for start in range(0, len(to_fetch), _BATCH_SIZE):
            batch_idx = to_fetch[start:start + _BATCH_SIZE]
            vectors = _embed_batch([texts[i] for i in batch_idx], api_key)
            for i, vec in zip(batch_idx, vectors):
                cache[hashes[i]] = vec
        if to_fetch:
            _save_cache(cache_path, cache)
    except urllib.error.HTTPError as e:
        return {"available": False, "reason": f"Embedding request failed: HTTP {e.code} {e.reason}"}
    except urllib.error.URLError as e:
        return {"available": False, "reason": f"Embedding request failed: {e.reason}"}
    except Exception as e:
        return {"available": False, "reason": f"Embedding request failed: {e}"}

    vectors = [cache[h] for h in hashes]
    n = len(nodes)
    dim = len(vectors[0])
    mean_vec = [sum(v[d] for v in vectors) / n for d in range(dim)]
    centered = [[v[d] - mean_vec[d] for d in range(dim)] for v in vectors]

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        matrix[i][i] = 1.0
        for j in range(i + 1, n):
            sim = cosine(centered[i], centered[j])
            matrix[i][j] = sim
            matrix[j][i] = sim

    return {
        "available": True,
        "model": EMBEDDING_MODEL,
        "centered": True,
        "chunked": False,
        "items": [{"idx": node["idx"], "iter": node["iter"]} for node in nodes],
        "matrix": matrix,
    }
