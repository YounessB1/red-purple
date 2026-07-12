"""Apply structured JSON patches to agent files dict."""


def apply_patches(files: dict[str, str], patches: list[dict]) -> tuple[dict[str, str], list[dict]]:
    """Apply a list of patches to agent files. Returns (updated_files, report).

    Patch format:
        {"op": "append",       "file": "prompt.md",          "content": "..."}
        {"op": "insert_after", "file": "AGENTS.md",           "target": "## Recon", "content": "..."}
        {"op": "replace",      "file": ".opencode/skills/sqli/SKILL.md","target": "old text",  "content": "new text"}
        {"op": "delete",       "file": "prompt.md",           "target": "outdated rule"}

    `file` is relative to workspace/agent/ (i.e. the key in the files dict).
    `target` must match verbatim; replace/delete silently skip if not found.
    insert_after falls back to append when target is not found.
    """
    updated = dict(files)
    report: list[dict] = []

    for i, patch in enumerate(patches):
        op = patch.get("op", "")
        file_key = patch.get("file", "")
        content = patch.get("content", "")
        target = patch.get("target", "")
        entry: dict = {"index": i, "op": op, "file": file_key, "status": "unknown"}

        try:
            text = updated.get(file_key, "")

            if op == "append":
                if text:
                    updated[file_key] = text.rstrip() + "\n\n" + content + "\n"
                else:
                    updated[file_key] = content.rstrip() + "\n"
                entry["status"] = "applied_append"

            elif op == "insert_after":
                if target and target in text:
                    idx = text.index(target) + len(target)
                    nl = text.find("\n", idx)
                    insert_at = (nl + 1) if nl != -1 else len(text)
                    updated[file_key] = text[:insert_at] + "\n" + content + "\n" + text[insert_at:]
                    entry["status"] = "applied_insert_after"
                else:
                    if text:
                        updated[file_key] = text.rstrip() + "\n\n" + content + "\n"
                    else:
                        updated[file_key] = content.rstrip() + "\n"
                    entry["status"] = "applied_insert_after_fallback_append"

            elif op == "replace":
                if target and target in text:
                    updated[file_key] = text.replace(target, content, 1)
                    entry["status"] = "applied_replace"
                else:
                    entry["status"] = "skipped_target_not_found"

            elif op == "delete":
                if not target:
                    entry["status"] = "skipped_no_target"
                elif target in text:
                    updated[file_key] = text.replace(target, "", 1)
                    entry["status"] = "applied_delete"
                else:
                    entry["status"] = "skipped_target_not_found"

            else:
                entry["status"] = f"skipped_unknown_op_{op}"

        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)

        report.append(entry)

    return updated, report
