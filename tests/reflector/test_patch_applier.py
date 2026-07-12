from source.optimize_anything.patch_applier import apply_patches


def test_append_new_skill_starts_with_frontmatter():
    content = "---\nname: sqli\ndescription: SQL injection guidance.\n---\n\n# SQLi\n"

    updated, report = apply_patches(
        {},
        [{"op": "append", "file": ".opencode/skills/sqli/SKILL.md", "content": content}],
    )

    assert report[0]["status"] == "applied_append"
    assert updated[".opencode/skills/sqli/SKILL.md"].startswith("---\nname: sqli\n")


def test_append_existing_file_keeps_separator():
    updated, report = apply_patches(
        {"AGENTS.md": "Existing rule.\n"},
        [{"op": "append", "file": "AGENTS.md", "content": "New rule."}],
    )

    assert report[0]["status"] == "applied_append"
    assert updated["AGENTS.md"] == "Existing rule.\n\nNew rule.\n"
