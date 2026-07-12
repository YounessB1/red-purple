#!/usr/bin/env python3
"""Create curated exploit-family transfer splits for XBOW.

The split design is intentionally based on the solution/exploit path rather than
only benchmark tags.  Train examples are chosen as reusable exploit modules;
validation examples exercise the same module on a different surface; the rest
are left as a stratified test set.

Outputs:
    source/dataset/splits15.json
    source/dataset/splits15_report.md
    source/dataset/split30.json
    source/dataset/split30_report.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOLUTIONS_INDEX = _REPO_ROOT / "xbow" / "solutions" / "index.json"
_FORBIDDEN_PATH = Path(__file__).parent / "forbidden_challenges.json"
_DEFAULT_OUTPUT_DIR = Path(__file__).parent

_EXCLUDED_BENCHMARKS: set[str] = set(
    json.loads(_FORBIDDEN_PATH.read_text(encoding="utf-8")).keys()
)


SPLITS15_TRAIN = [
    "XBEN-001-24",
    "XBEN-005-24",
    "XBEN-007-24",
    "XBEN-009-24",
    "XBEN-016-24",
    "XBEN-023-24",
    "XBEN-033-24",
    "XBEN-039-24",
    "XBEN-040-24",
    "XBEN-041-24",
    "XBEN-061-24",
    "XBEN-068-24",
    "XBEN-075-24",
    "XBEN-091-24",
    "XBEN-018-24",
]

SPLITS15_VAL = [
    "XBEN-008-24",
    "XBEN-049-24",
    "XBEN-021-24",
    "XBEN-024-24",
    "XBEN-028-24",
    "XBEN-043-24",
    "XBEN-044-24",
    "XBEN-052-24",
    "XBEN-073-24",
    "XBEN-078-24",
    "XBEN-079-24",
    "XBEN-081-24",
    "XBEN-089-24",
    "XBEN-093-24",
    "XBEN-100-24",
]

SPLIT30_TRAIN_ADDITIONS = [
    "XBEN-003-24",
    "XBEN-011-24",
    "XBEN-022-24",
    "XBEN-026-24",
    "XBEN-048-24",
    "XBEN-038-24",
    "XBEN-045-24",
    "XBEN-053-24",
    "XBEN-057-24",
    "XBEN-058-24",
    "XBEN-060-24",
    "XBEN-067-24",
    "XBEN-072-24",
    "XBEN-082-24",
    "XBEN-096-24",
]

SPLIT30_VAL_ADDITIONS = [
    "XBEN-002-24",
    "XBEN-027-24",
    "XBEN-030-24",
    "XBEN-032-24",
    "XBEN-037-24",
    "XBEN-004-24",
    "XBEN-046-24",
    "XBEN-050-24",
    "XBEN-063-24",
    "XBEN-071-24",
    "XBEN-076-24",
    "XBEN-085-24",
    "XBEN-101-24",
    "XBEN-095-24",
    "XBEN-097-24",
]


FAMILY_BY_ID = {
    "XBEN-001-24": "idor_order_receipt",
    "XBEN-002-24": "idor_order_receipt",
    "XBEN-003-24": "idor_header_identity",
    "XBEN-004-24": "xss_event_filter_bypass",
    "XBEN-005-24": "auth_mass_assignment",
    "XBEN-006-24": "sqli_filter_bypass",
    "XBEN-007-24": "graphql_schema_idor",
    "XBEN-008-24": "xss_event_filter_bypass",
    "XBEN-009-24": "ssti_context_leak",
    "XBEN-011-24": "xss_event_filter_bypass",
    "XBEN-012-24": "xss_attribute_breakout",
    "XBEN-014-24": "deserialization_auth_type_juggling",
    "XBEN-015-24": "xss_event_filter_bypass",
    "XBEN-016-24": "xss_event_filter_bypass",
    "XBEN-017-24": "xss_attribute_breakout",
    "XBEN-018-24": "xss_filter_reasoning",
    "XBEN-019-24": "file_read_lfi",
    "XBEN-020-24": "ssrf_internal_fetch",
    "XBEN-021-24": "idor_client_state",
    "XBEN-022-24": "ssti_upload_include",
    "XBEN-023-24": "ssti_blind_object_traversal",
    "XBEN-024-24": "ssrf_internal_action",
    "XBEN-025-24": "ssti_context_leak",
    "XBEN-026-24": "apache_path_traversal_rce",
    "XBEN-027-24": "jwt_identity_confusion",
    "XBEN-028-24": "upload_lfi_chain",
    "XBEN-029-24": "sqli_then_upload_rce",
    "XBEN-030-24": "known_cve_php_rce",
    "XBEN-031-24": "apache_path_traversal_rce",
    "XBEN-032-24": "xxe_file_read",
    "XBEN-033-24": "ssrf_internal_fetch",
    "XBEN-034-24": "php_wrapper_rce",
    "XBEN-035-24": "framework_expression_rce",
    "XBEN-036-24": "pickle_deserialization",
    "XBEN-037-24": "blind_command_injection",
    "XBEN-038-24": "graphql_sqli",
    "XBEN-039-24": "sqli_auth_bypass",
    "XBEN-040-24": "sqli_credential_exfil",
    "XBEN-041-24": "upload_webshell_rce",
    "XBEN-042-24": "source_credential_harvest",
    "XBEN-043-24": "idor_hidden_identity",
    "XBEN-044-24": "ssti_engine_transfer",
    "XBEN-045-24": "command_injection_output_parser",
    "XBEN-046-24": "xss_event_filter_bypass",
    "XBEN-047-24": "xss_basic_reflection",
    "XBEN-048-24": "xss_attribute_breakout",
    "XBEN-049-24": "xss_js_filter_bypass",
    "XBEN-050-24": "xss_body_tag_bypass",
    "XBEN-051-24": "xss_attribute_breakout",
    "XBEN-052-24": "auth_hidden_parameter",
    "XBEN-053-24": "ssti_python_rce",
    "XBEN-054-24": "auth_method_tamper",
    "XBEN-055-24": "auth_type_juggling",
    "XBEN-057-24": "pickle_deserialization",
    "XBEN-058-24": "idor_numeric_resource",
    "XBEN-059-24": "xss_js_context_breakout",
    "XBEN-060-24": "auth_magic_hash",
    "XBEN-061-24": "lfi_log_poisoning",
    "XBEN-062-24": "xss_event_filter_bypass",
    "XBEN-063-24": "ssti_filtered_json",
    "XBEN-064-24": "xss_attribute_breakout",
    "XBEN-065-24": "xss_event_filter_bypass",
    "XBEN-066-24": "request_smuggling_auth",
    "XBEN-067-24": "upload_polyglot_rce",
    "XBEN-068-24": "command_injection_argument",
    "XBEN-069-24": "idor_password_reset",
    "XBEN-070-24": "xss_basic_reflection",
    "XBEN-071-24": "sqli_error_exfil",
    "XBEN-072-24": "source_credential_harvest",
    "XBEN-073-24": "command_injection_ping",
    "XBEN-074-24": "xss_keyword_filter",
    "XBEN-075-24": "yaml_deserialization",
    "XBEN-076-24": "ssti_python_rce",
    "XBEN-077-24": "xxe_svg_upload",
    "XBEN-078-24": "sqli_credential_exfil",
    "XBEN-079-24": "lfi_log_poisoning",
    "XBEN-080-24": "ssti_python_rce",
    "XBEN-081-24": "php_deserialization_cookie",
    "XBEN-082-24": "internal_api_command_execution",
    "XBEN-083-24": "sqli_hidden_category",
    "XBEN-084-24": "exposed_storage_credential_chain",
    "XBEN-085-24": "auth_ip_spoofing",
    "XBEN-086-24": "ssti_erb_rce",
    "XBEN-088-24": "race_condition_auth",
    "XBEN-089-24": "auth_role_parameter",
    "XBEN-090-24": "idor_admin_drafts",
    "XBEN-091-24": "jwt_alg_none",
    "XBEN-093-24": "ssti_python_rce",
    "XBEN-094-24": "command_injection_file_exfil",
    "XBEN-095-24": "blind_sqli_credential_exfil",
    "XBEN-096-24": "xxe_file_read",
    "XBEN-097-24": "nginx_alias_traversal",
    "XBEN-098-24": "file_read_lfi",
    "XBEN-099-24": "idor_predictable_object_id",
    "XBEN-100-24": "graphql_nosqli",
    "XBEN-101-24": "crypto_padding_oracle",
    "XBEN-102-24": "idor_profile_update",
    "XBEN-103-24": "crypto_cookie_identity",
    "XBEN-104-24": "ssti_python_rce",
}


TRAIN_REASONS = {
    "XBEN-001-24": "Canonical authenticated IDOR on adjacent order receipts; teaches numeric object enumeration after login.",
    "XBEN-005-24": "Mass-assignment privilege flip in profile update; teaches inspection of hidden/trusted POST fields.",
    "XBEN-007-24": "GraphQL IDOR plus introspection; teaches reading API schema before probing fields and identities.",
    "XBEN-009-24": "Django SSTI context leak; teaches cheap arithmetic/template probes and using debug/context clues.",
    "XBEN-016-24": "Reflected XSS with blocked script, spaces, and quotes; teaches event-handler and separator bypass reasoning.",
    "XBEN-018-24": "Hard reflected XSS with non-obvious tag filtering; teaches systematic filter analysis under level-3 difficulty.",
    "XBEN-023-24": "Blind Python template/eval path; teaches object traversal and boolean exfiltration when output is indirect.",
    "XBEN-033-24": "SSRF through a converter URL; teaches internal route discovery after confirming a fetch primitive.",
    "XBEN-039-24": "Simple SQLi authentication bypass; teaches probing auth boundaries before complex exploitation.",
    "XBEN-040-24": "Blind SQLi to credential extraction and admin login; teaches exfiltrate-then-chain workflow.",
    "XBEN-041-24": "Simple upload-to-webshell RCE; teaches validating upload execution path before chaining it.",
    "XBEN-061-24": "LFI escalated through access-log poisoning; teaches file-read primitive to execution pivot.",
    "XBEN-068-24": "Argument-style command injection; teaches shell-free and parser-aware command execution thinking.",
    "XBEN-075-24": "YAML cookie deserialization; teaches spotting structured client state that is executed server-side.",
    "XBEN-091-24": "JWT alg=none privilege escalation; teaches token structure tampering and unsigned identity claims.",
    "XBEN-101-24": "Padding oracle against encrypted captcha cookie; teaches oracle-driven state recovery for hard crypto tests.",
    "XBEN-003-24": "Header-controlled identity IDOR; broadens IDOR training beyond URL/path parameters.",
    "XBEN-011-24": "POST-based unclosed SVG XSS; reinforces event-handler payloads on a different input method.",
    "XBEN-022-24": "Uploaded Jinja payload included through path traversal; trains the upload plus template execution bridge.",
    "XBEN-026-24": "Apache 2.4.50 traversal/RCE; teaches version-aware CVE exploitation and encoded traversal checks.",
    "XBEN-031-24": "Apache 2.4.49 CGI traversal/RCE; pairs with XBEN-026 to teach version-specific path normalization failures.",
    "XBEN-038-24": "GraphQL resolver SQLi; trains schema discovery followed by injection in API arguments.",
    "XBEN-045-24": "Command injection with output parser constraints; trains adapting payload shape to application parsers.",
    "XBEN-048-24": "Attribute-context reflected XSS; broadens XSS training beyond event tags into quote/attribute breakout reasoning.",
    "XBEN-053-24": "Direct Jinja RCE via object traversal; strengthens the Python SSTI path used by several held-out cases.",
    "XBEN-057-24": "Pickle deserialization with state manipulation; adds Python-specific unsafe deserialization coverage.",
    "XBEN-058-24": "Small numeric IDOR on company jobs; provides the simplest version of authenticated object swapping.",
    "XBEN-060-24": "Magic-hash type juggling; teaches weak comparison and hash-looking value pitfalls in auth.",
    "XBEN-067-24": "Polyglot/double-extension upload RCE; extends upload training beyond a trivial webshell.",
    "XBEN-072-24": "Hardcoded client-side admin credentials; teaches source inspection before exploit escalation.",
    "XBEN-082-24": "Internal API command execution via JSON and proxy path behavior; trains cross-service primitive chaining.",
    "XBEN-096-24": "Plain XXE file read; trains protocol-aware XML parser testing.",
}


VAL_REASONS = {
    "XBEN-008-24": "Validation for XSS event-handler transfer: same reflected-input habit as XBEN-016, different allowed tag/event.",
    "XBEN-018-24": "Hard XSS validation: tests whether filter analysis from XBEN-016 generalizes to non-obvious tag rules.",
    "XBEN-021-24": "Validation for client-state IDOR: transfers IDOR enumeration and token inspection from XBEN-001 and XBEN-091.",
    "XBEN-024-24": "Validation for SSRF pivoting: applies XBEN-033 internal-fetch habit to a state-changing localhost action.",
    "XBEN-028-24": "Validation for primitive chaining: combines upload knowledge from XBEN-041 with inclusion/LFI reasoning from XBEN-061.",
    "XBEN-043-24": "Validation for hidden identity selectors: transfers IDOR enumeration from URL/header surfaces to login POST state.",
    "XBEN-044-24": "Validation for SSTI engine transfer: uses probe-confirm-escalate habits from Django/Python SSTI on Twig.",
    "XBEN-052-24": "Validation for hidden trust parameters: tests mass-assignment lessons from XBEN-005 on a simpler auth form.",
    "XBEN-073-24": "Validation for command injection: transfers parser/argument injection thinking from XBEN-068 to a ping endpoint.",
    "XBEN-078-24": "Validation for SQLi credential chaining: applies XBEN-039/XBEN-040 lessons to category UNION exfiltration.",
    "XBEN-079-24": "Validation for LFI log poisoning: same exploit family as XBEN-061 with different traversal/path constraints.",
    "XBEN-080-24": "Validation for Python SSTI RCE: applies object traversal habits from XBEN-023 to a form workflow.",
    "XBEN-081-24": "Validation for deserialization: transfers unsafe client-state reasoning from YAML/pickle-like training to PHP serialization.",
    "XBEN-089-24": "Validation for auth trust-boundary probing: applies hidden role and token-tampering habits to login role selection plus static 2FA.",
    "XBEN-100-24": "Validation for API-schema transfer: extends GraphQL schema reading from XBEN-007 to NoSQL operator injection.",
    "XBEN-004-24": "Validation for XSS filter bypass: tests whether event/filter reasoning transfers to the Cross 25 rule set.",
    "XBEN-002-24": "Validation for IDOR object enumeration: same trading-platform receipt family as XBEN-001, held out for indirect IDs.",
    "XBEN-027-24": "Validation for mixed session/JWT identity confusion: combines IDOR/user discovery with XBEN-091 alg=none training.",
    "XBEN-030-24": "Validation for known-CVE RCE: tests version and wrapper reasoning after Apache CVE/path traversal training.",
    "XBEN-032-24": "Validation for XML protocol probing: applies XBEN-096 XXE habits to SOAP.",
    "XBEN-037-24": "Validation for blind command injection: applies command-injection confirmation from XBEN-045/XBEN-068 without direct output.",
    "XBEN-046-24": "Validation for XSS SVG/event bypass: tests the same event-handler family after additional XBEN-011 training.",
    "XBEN-049-24": "Validation for XSS filter bypass in JavaScript/string construction: checks whether payload construction generalizes beyond raw events.",
    "XBEN-050-24": "Validation for XSS tag allowlist reasoning: tests body-tag/event execution after attribute and event-handler training.",
    "XBEN-063-24": "Validation for filtered Jinja in JSON construction: applies SSTI traversal from XBEN-022/XBEN-023/XBEN-053 under character filters.",
    "XBEN-071-24": "Validation for SQLi discovery and dump: tests SQLi workflow from XBEN-038/XBEN-039 on a contact form.",
    "XBEN-076-24": "Validation for direct Jinja RCE: a simpler held-out target for XBEN-053 and XBEN-023 SSTI traversal habits.",
    "XBEN-077-24": "Validation for uploaded XML/SVG parsing: transfers XBEN-096 XXE file-read knowledge to upload handling.",
    "XBEN-085-24": "Validation for auth boundary spoofing: applies source/header trust lessons to X-Forwarded-For based access.",
    "XBEN-093-24": "Hard validation for Flask/Jinja SSTI: tests whether the training handles route mapping and suspicious parameter escalation.",
    "XBEN-101-24": "Hard validation for crypto/state recovery: tests whether oracle-style reasoning transfers from auth and token-state training.",
    "XBEN-095-24": "Validation for blind SQLi credential extraction: held out after SQLi auth/exfiltration training.",
    "XBEN-097-24": "Validation for path normalization bugs: applies Apache traversal lessons to Nginx alias traversal.",
}


FAMILY_TRANSFER = {
    "idor_order_receipt": "IDOR object enumeration after low-privilege login.",
    "idor_header_identity": "identity-selector probing from IDOR and auth-state training.",
    "auth_mass_assignment": "hidden/trusted parameter inspection in account and profile updates.",
    "graphql_schema_idor": "schema-first API probing before field or identity manipulation.",
    "graphql_sqli": "GraphQL schema reading plus SQLi workflow transfer.",
    "graphql_nosqli": "schema-first API probing plus operator-injection transfer from SQLi modules.",
    "xss_event_filter_bypass": "XSS event-handler and filter bypass training.",
    "xss_attribute_breakout": "reflected attribute-context reasoning from XSS filter training.",
    "xss_filter_reasoning": "systematic filter analysis from the XSS training set.",
    "xss_js_filter_bypass": "payload construction and string/filter reasoning from XSS training.",
    "xss_js_context_breakout": "JavaScript-context breakout reasoning from XSS training.",
    "xss_basic_reflection": "basic reflected-XSS confirmation from the event/filter XSS module.",
    "xss_body_tag_bypass": "tag allowlist bypass reasoning from the XSS module.",
    "xss_keyword_filter": "keyword-filter bypass reasoning from the XSS module.",
    "ssti_context_leak": "cheap SSTI probes and context/error analysis.",
    "ssti_blind_object_traversal": "blind SSTI and Python object traversal.",
    "ssti_upload_include": "upload plus template/file inclusion bridging.",
    "ssti_python_rce": "Python object traversal and probe-confirm-escalate SSTI workflow.",
    "ssti_filtered_json": "filtered SSTI traversal under JSON/string-construction constraints.",
    "ssti_engine_transfer": "engine-agnostic SSTI probing and escalation.",
    "ssti_erb_rce": "template-engine transfer from the SSTI module.",
    "ssrf_internal_fetch": "internal fetch probing through user-controlled URL surfaces.",
    "ssrf_internal_action": "SSRF pivoting from internal fetch to state-changing endpoint.",
    "sqli_auth_bypass": "auth-boundary SQLi probing.",
    "sqli_credential_exfil": "credential extraction and admin-login chaining.",
    "sqli_filter_bypass": "SQLi confirmation plus filter/allowlist reasoning.",
    "sqli_error_exfil": "SQLi discovery and exfiltration workflow.",
    "sqli_hidden_category": "SQLi category/filter reasoning from the SQLi module.",
    "blind_sqli_credential_exfil": "blind SQLi extraction and credential verification.",
    "sqli_then_upload_rce": "SQLi credential chaining plus upload execution.",
    "upload_webshell_rce": "upload execution validation.",
    "upload_polyglot_rce": "upload validation with extension/content bypasses.",
    "upload_lfi_chain": "upload execution and LFI chaining.",
    "lfi_log_poisoning": "LFI-to-RCE escalation through poisoned server-controlled files.",
    "file_read_lfi": "file-read primitive discovery and immediate exfiltration.",
    "apache_path_traversal_rce": "path normalization/CVE reasoning from Apache training.",
    "nginx_alias_traversal": "path traversal transfer from Apache traversal to Nginx alias behavior.",
    "php_wrapper_rce": "file wrapper/RCE reasoning from path traversal and known-CVE modules.",
    "known_cve_php_rce": "version-aware exploitation and wrapper reasoning from CVE/path traversal training.",
    "xxe_file_read": "XML external entity probing and local file read validation.",
    "xxe_svg_upload": "XXE file-read transfer to SVG upload parsing.",
    "command_injection_argument": "command injection confirmation under argument/parser constraints.",
    "command_injection_ping": "command injection transfer to network utility wrappers.",
    "command_injection_output_parser": "parser-aware command injection.",
    "blind_command_injection": "time/side-channel command-injection validation from command injection training.",
    "command_injection_file_exfil": "command execution plus file exfiltration from command injection modules.",
    "framework_expression_rce": "framework-specific expression execution after command/SSTI training.",
    "internal_api_command_execution": "cross-service primitive chaining.",
    "yaml_deserialization": "unsafe deserialization of structured client state.",
    "pickle_deserialization": "unsafe Python deserialization and state manipulation.",
    "php_deserialization_cookie": "client-state deserialization transfer across formats.",
    "deserialization_auth_type_juggling": "serialized auth state plus loose comparison reasoning.",
    "auth_hidden_parameter": "hidden trust-field inspection.",
    "auth_role_parameter": "role/trust-boundary probing plus token/state tampering.",
    "auth_method_tamper": "auth-boundary probing and direct restricted-endpoint checks.",
    "auth_type_juggling": "loose comparison reasoning in authentication logic.",
    "auth_magic_hash": "weak hash comparison reasoning in authentication logic.",
    "auth_ip_spoofing": "trusted-header/source-boundary probing from auth and IDOR modules.",
    "jwt_alg_none": "JWT structure and signature-trust tampering.",
    "jwt_identity_confusion": "JWT/session identity transfer plus IDOR enumeration.",
    "idor_client_state": "client-state identity decoding and enumeration.",
    "idor_hidden_identity": "identity selector enumeration across hidden, header, and URL surfaces.",
    "idor_numeric_resource": "numeric resource enumeration after authentication.",
    "idor_admin_drafts": "IDOR object enumeration after low-privilege login.",
    "idor_password_reset": "IDOR identity mutation and account-takeover chaining.",
    "idor_profile_update": "profile identity mutation from mass-assignment/IDOR training.",
    "idor_predictable_object_id": "object identifier reasoning after IDOR enumeration training.",
    "source_credential_harvest": "source inspection and credential chaining.",
    "exposed_storage_credential_chain": "credential harvest plus session use from source/API modules.",
    "crypto_padding_oracle": "oracle-driven state recovery.",
    "crypto_cookie_identity": "token/cookie state reasoning across auth and crypto surfaces.",
    "request_smuggling_auth": "advanced auth-boundary stress case after header/session trust training.",
    "race_condition_auth": "advanced auth-boundary stress case after state-transition training.",
}


def _load_records() -> dict[str, dict]:
    index = json.loads(_SOLUTIONS_INDEX.read_text(encoding="utf-8"))
    records: dict[str, dict] = {}
    for solution in index["solutions"]:
        bench_id = solution["benchmark_id"]
        if bench_id in _EXCLUDED_BENCHMARKS:
            continue
        records[bench_id] = {
            "id": bench_id,
            "level": int(solution["level"]),
            "tags": solution.get("tags", []),
            "name": solution.get("benchmark_name", bench_id),
            "solution_file": solution.get("solution_file"),
            "family": FAMILY_BY_ID.get(bench_id, "other"),
        }
    return records


def _level_counts(records: dict[str, dict], ids: list[str]) -> dict[str, int]:
    counts = Counter(str(records[bench_id]["level"]) for bench_id in ids)
    return {level: counts.get(level, 0) for level in ["1", "2", "3"]}


def _family_counts(records: dict[str, dict], ids: list[str]) -> dict[str, int]:
    counts = Counter(records[bench_id]["family"] for bench_id in ids)
    return dict(sorted(counts.items()))


def _reason_for(bench_id: str, split: str) -> str:
    if split == "train":
        return TRAIN_REASONS.get(
            bench_id,
            f"Selected for training as a representative of {FAMILY_BY_ID.get(bench_id, 'its exploit family')}.",
        )
    if split == "val":
        return VAL_REASONS.get(
            bench_id,
            f"Selected for validation to test transfer for {FAMILY_BY_ID.get(bench_id, 'its exploit family')}.",
        )
    family = FAMILY_BY_ID.get(bench_id, "other")
    transfer = FAMILY_TRANSFER.get(family, "general reconnaissance and exploit-chaining habits from training.")
    return f"Left in test to measure held-out transfer for {family}; expected useful training: {transfer}"


def _build_split(
    name: str,
    train: list[str],
    val: list[str],
    records: dict[str, dict],
    extends: str | None = None,
) -> dict:
    all_ids = sorted(records)
    test = sorted(set(all_ids) - set(train) - set(val))

    assignments = {}
    for split_name, ids in [("train", train), ("val", val), ("test", test)]:
        for bench_id in ids:
            record = records[bench_id]
            assignments[bench_id] = {
                "split": split_name,
                "level": record["level"],
                "family": record["family"],
                "tags": record["tags"],
                "name": record["name"],
                "reason": _reason_for(bench_id, split_name),
                "solution_file": record["solution_file"],
            }

    return {
        "_meta": {
            "strategy": "curated_exploit_family_stratified_transfer",
            "generated_by": "source/dataset/transfer_splits.py",
            "source": "xbow/solutions/index.json",
            "extends": extends,
            "goal": (
                "Train on reusable exploit workflows, validate on directly related but held-out "
                "surfaces, and leave the rest as a broad stratified test set."
            ),
            "forbidden_excluded": sorted(_EXCLUDED_BENCHMARKS),
            "total_usable": len(all_ids),
            "split_counts": {
                "train": len(train),
                "val": len(val),
                "test": len(test),
            },
            "level_counts": {
                "train": _level_counts(records, train),
                "val": _level_counts(records, val),
                "test": _level_counts(records, test),
            },
            "family_counts": {
                "train": _family_counts(records, train),
                "val": _family_counts(records, val),
                "test": _family_counts(records, test),
            },
            "design_notes": [
                "Forbidden challenges are excluded using source/dataset/forbidden_challenges.json.",
                "Families are assigned from solution workflows, not just benchmark tags.",
                "split30 strictly extends splits15: train and validation each keep all splits15 members.",
                "Test is the left-out remainder, used for broader held-out transfer checks.",
            ],
            "assignments": dict(sorted(assignments.items())),
        },
        "train": list(train),
        "val": list(val),
        "test": test,
    }


def _validate_split(name: str, split: dict, records: dict[str, dict]) -> None:
    train = split["train"]
    val = split["val"]
    test = split["test"]
    all_assigned = train + val + test
    expected = set(records)
    if set(all_assigned) != expected:
        missing = sorted(expected - set(all_assigned))
        extra = sorted(set(all_assigned) - expected)
        raise RuntimeError(f"{name}: assignment mismatch, missing={missing}, extra={extra}")
    if len(all_assigned) != len(set(all_assigned)):
        dupes = [bench_id for bench_id, count in Counter(all_assigned).items() if count > 1]
        raise RuntimeError(f"{name}: duplicate assignments: {sorted(dupes)}")
    forbidden = sorted(set(all_assigned) & _EXCLUDED_BENCHMARKS)
    if forbidden:
        raise RuntimeError(f"{name}: forbidden benchmarks included: {forbidden}")


def _validate_extension(splits15: dict, split30: dict) -> None:
    missing_train = sorted(set(splits15["train"]) - set(split30["train"]))
    missing_val = sorted(set(splits15["val"]) - set(split30["val"]))
    if missing_train or missing_val:
        raise RuntimeError(
            f"split30 must extend splits15; missing_train={missing_train}, missing_val={missing_val}"
        )


def _render_report(name: str, split: dict) -> str:
    meta = split["_meta"]
    assignments = meta["assignments"]
    lines = [
        f"# {name} Report",
        "",
        "## Purpose",
        "",
        meta["goal"],
        "",
        "## Counts",
        "",
        f"- Usable challenges: {meta['total_usable']}",
        f"- Train: {meta['split_counts']['train']}  Level mix: {meta['level_counts']['train']}",
        f"- Val: {meta['split_counts']['val']}  Level mix: {meta['level_counts']['val']}",
        f"- Test: {meta['split_counts']['test']}  Level mix: {meta['level_counts']['test']}",
        f"- Extends: {meta['extends'] or 'none'}",
        f"- Forbidden excluded: {', '.join(meta['forbidden_excluded'])}",
        "",
        "## Design",
        "",
        "- Stratification is based on exploit workflow families extracted from solutions.",
        "- Train contains reusable primitives; val contains near-neighbor transfer cases.",
        "- Test contains the left-out remainder and keeps broad family/difficulty coverage.",
        "",
    ]

    for split_name in ["train", "val", "test"]:
        title = {"train": "Training", "val": "Validation", "test": "Test"}[split_name]
        lines.extend([f"## {title} Allocation", ""])
        for bench_id in split[split_name]:
            item = assignments[bench_id]
            tags = ", ".join(item["tags"])
            lines.append(
                f"- {bench_id} (L{item['level']}, {item['family']}, tags: {tags}): {item['reason']}"
            )
        lines.append("")

    lines.extend(
        [
            "## Transfer Summary",
            "",
            "The main transfer paths are:",
            "",
            "- IDOR/auth state: train on numeric IDs, hidden trust fields, JWTs, and client state; validate/test on hidden IDs, cookies, role parameters, and mixed session/JWT behavior.",
            "- XSS: train on event-handler and filter bypasses; validate/test on different tag allowlists, JavaScript/string contexts, and harder filter rules.",
            "- SSTI: train on context leak, blind object traversal, and Jinja/Python RCE; validate/test on Twig, ERB, filtered JSON, and route-specific template sinks.",
            "- SQL/API injection: train on auth bypass, blind credential exfiltration, and GraphQL schema reading; validate/test on category filters, GraphQL/NoSQL operators, and hidden tables.",
            "- File, SSRF, XXE, upload, and traversal: train on internal fetches, upload execution, LFI/log poisoning, path traversal, and XXE; validate/test on combined chains and alternate server stacks.",
            "- RCE/deserialization/crypto: train on unsafe structured state, command injection, and oracle-style token recovery; validate/test on different serialization formats, service boundaries, and state machines.",
            "",
        ]
    )
    return "\n".join(lines)


def create_outputs(output_dir: Path) -> None:
    records = _load_records()
    if sorted(FAMILY_BY_ID) != sorted(records):
        missing = sorted(set(records) - set(FAMILY_BY_ID))
        stale = sorted(set(FAMILY_BY_ID) - set(records))
        raise RuntimeError(f"family map mismatch, missing={missing}, stale={stale}")

    splits15 = _build_split(
        "splits15",
        SPLITS15_TRAIN,
        SPLITS15_VAL,
        records,
    )
    split30 = _build_split(
        "split30",
        SPLITS15_TRAIN + SPLIT30_TRAIN_ADDITIONS,
        SPLITS15_VAL + SPLIT30_VAL_ADDITIONS,
        records,
        extends="splits15",
    )

    _validate_split("splits15", splits15, records)
    _validate_split("split30", split30, records)
    _validate_extension(splits15, split30)

    outputs = [
        ("splits15.json", json.dumps(splits15, indent=2) + "\n"),
        ("split30.json", json.dumps(split30, indent=2) + "\n"),
        ("splits15_report.md", _render_report("splits15", splits15)),
        ("split30_report.md", _render_report("split30", split30)),
    ]
    for filename, content in outputs:
        (output_dir / filename).write_text(content, encoding="utf-8")

    for name, split in [("splits15", splits15), ("split30", split30)]:
        counts = split["_meta"]["split_counts"]
        levels = split["_meta"]["level_counts"]
        print(
            f"{name}: train={counts['train']} val={counts['val']} test={counts['test']} "
            f"levels(train={levels['train']}, val={levels['val']}, test={levels['test']})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    create_outputs(args.output_dir)


if __name__ == "__main__":
    main()
