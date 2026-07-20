"""Classify XBOW benchmark tags into human-readable vulnerability categories."""
import json
from pathlib import Path

TAG_TO_CATEGORY: dict[str, str] = {
    "xss": "XSS", "ssti": "SSTI", "idor": "IDOR",
    "privilege_escalation": "Privilege Escalation", "command_injection": "Command Injection",
    "ssh": "Command Injection", "default_credentials": "Default Credentials",
    "sqli": "SQL Injection", "blind_sqli": "SQL Injection", "nosqli": "SQL Injection",
    "insecure_deserialization": "Deserialization", "lfi": "LFI",
    "path_traversal": "LFI", "xxe": "LFI", "business_logic": "Business Logic",
    "graphql": "Business Logic", "race_condition": "Business Logic",
    "brute_force": "Business Logic", "http_method_tamper": "Business Logic",
    "smuggling_desync": "Business Logic", "crypto": "Business Logic",
    "jwt": "Business Logic", "ssrf": "Business Logic",
    "information_disclosure": "Business Logic", "arbitrary_file_upload": "Business Logic",
    "cve": "Business Logic",
}
CATEGORY_PRIORITY = [
    "XSS", "SSTI", "SQL Injection", "Command Injection",
    "Deserialization", "LFI", "IDOR", "Privilege Escalation",
    "Default Credentials", "Business Logic",
]


def _classify_tags(tags: list) -> str:
    cats = {TAG_TO_CATEGORY[t] for t in tags if t in TAG_TO_CATEGORY}
    if not cats:
        return "Business Logic"
    return min(cats, key=CATEGORY_PRIORITY.index)


def load_bench_categories() -> dict:
    # parents[3]: visualize/ -> result-analysis/ -> source/ -> repo root
    benchmarks_dir = Path(__file__).resolve().parents[3] / "xbow" / "benchmarks"
    result: dict[str, str] = {}
    if not benchmarks_dir.exists():
        return result
    for bench_dir in sorted(benchmarks_dir.iterdir()):
        bj = bench_dir / "benchmark.json"
        if not bj.exists():
            continue
        try:
            data = json.loads(bj.read_text(encoding="utf-8"))
            result[bench_dir.name] = _classify_tags(data.get("tags", []))
        except Exception:
            pass
    return result
