# Cross-target heuristics

- Always view page source, HTML comments, and client-side JS before attacking — hardcoded/leaked credentials show up there constantly.
- Try default credentials (`test:test`, `admin:admin`, `user:user`) before anything else.
- Any numeric or encoded ID (URL path, query param, header, hidden field, cookie) is fuzzable — try adjacent and unrelated values.
- Deliberately try to trigger an error or debug page (bad input, wrong content-type, missing param) — it often leaks the stack, template context, or config.
- Enumerate every discovered port/service, not just the main web app — secondary services (storage, internal APIs) are sometimes the real target.
- Confirm blind vulnerabilities (SSTI/SQLi/command injection) with a timing or boolean signal before concluding a technique failed.
- Filters are usually incomplete blocklists, not full sanitization — probe which specific characters/tags/keywords are blocked, then find the smallest bypass, rather than switching vulnerability class.
- A real flag only counts when it comes from target output (HTTP response, file read, command output) — never report a guessed or self-generated string.
