Cross-target tradecraft (keep it lean):

- XSS efficiency: First identify the reflection context (inside attribute value, in element body, inside JS, or after a closed tag). In "after-tag" contexts you must create a new element that auto-fires (e.g., <input autofocus onfocus=alert`1`>) rather than adding bare attributes.
- XSS filter probing: Test case sensitivity and keyword scope. Prefer quote/paren-free payloads (alert`1`) and non-blacklisted events (onfocus with autofocus, onanimationstart with injected CSS) before heavier obfuscation.
- Exposed repos: If /.git is reachable, prioritize dumping it (HEAD, refs, packed-refs, index, objects). If a front controller intervenes, try path rewrites/encodings (/.git/HEAD, /.git/./HEAD, .git%2FHEAD, //, trailing dot or semicolon) until raw files are retrievable.
- High-signal leads first: Concrete indicators (debug traces, secret files, backup archives, repo leaks) outrank speculative attacks (blind SQLi, generic fuzzing). Exploit the clearest lead before exploring others.
- Router interference: When content looks proxied by a front controller, attempt direct static file fetch with encodings/double slashes/segment tricks before giving up.
- Stop early when flag is obtained; avoid extra enumeration that risks timeouts.
