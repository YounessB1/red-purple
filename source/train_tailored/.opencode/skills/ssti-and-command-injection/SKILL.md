---
name: ssti-and-command-injection
description: Use when user input is reflected into a template/rendered output, or when an endpoint clearly shells out to a system command (ping, convert, etc.).
---

# Server-side template injection and command injection

Both are "breakout from an interpreter" bugs — same recon posture: probe
with something evaluable, watch for evaluation vs literal reflection.

## SSTI

### 1. Probe and identify the engine
- Send `{{7*7}}` in any reflected input. If it renders as `49`, you have
  Jinja2/Django/Twig-family SSTI. For ERB (Ruby) try `<%= 7*8 %>` → `56`.
- Even a `TemplateSyntaxError` (rather than silent literal reflection)
  confirms the input reaches a template engine — read the error page
  carefully, it sometimes reveals the full template context, including a
  `FLAG`/`flag` variable that may be directly renderable, e.g.
  `{{ FLAG }}` or `{{ flag }}`, with no further injection needed.

### 2. If blacklisted/sandboxed
Common blocked characters: `{`, `}`, `.`, `_`, `[`, `]`. Work around with:
- Tag forms instead of expression forms: `{% print ... %}`, `{% if ... %}`
- Attribute-access filters instead of dots: `|attr('__class__')` in place
  of `.__class__`
- Build forbidden substrings from allowed pieces via string concatenation
  or by pulling them from an allowed source, e.g.
  `request.args.get('x')` (invoked via `|attr()`) to smuggle in strings
  like `__class__`, `__builtins__` through a query parameter instead of
  typing them directly in the blocked payload.

### 3. Escalate to RCE (Jinja2/Python)
Classic class-traversal ladder to reach `__builtins__`/`subprocess` from
an object you can already reach (e.g. `""`, `request`, a form field):
```
().__class__.__base__.__subclasses__()
```
Walk the resulting list for a subclass exposing `__init__.__globals__`
(e.g. `warnings.catch_warnings`), from there reach `__builtins__`, then
`__import__('os').popen('cmd').read()` or `subprocess.Popen(...)`.
For Twig (PHP), look for known sandbox-bypass gadgets for the specific
version (e.g. `registerUndefinedFilterCallback`). For ERB, `<%= `cmd` %>`
(backtick shell-out) often works directly once injection is confirmed.

### 4. Blind SSTI (output not reflected)
If the template result affects response *shape* rather than visible text
(e.g. number of rows returned, redirect vs error), encode a boolean
comparison of one target byte at a time and binary-search the value —
identical technique to blind SQLi. This is slow; script it.

## Command injection

### 1. Confirm, including blind
- Standard separators: `; `, `&&`, `||`, backticks, `$(...)`.
- If no output is reflected, confirm with a **timing** signal:
  `; sleep 5` (or `| sleep 5`) and measure response delay before
  concluding the endpoint isn't injectable.

### 2. Handle filtering/parsing
- If spaces are filtered, use `${IFS}` or input redirection `<` in place
  of a space.
- If output is filtered/parsed (e.g. the app expects "ping"-shaped output
  and strips anything else), craft your payload's output to mimic the
  expected format so the parser doesn't discard it, or exfiltrate via a
  side channel (write to a reachable static file, trigger a timing
  difference, or use DNS/HTTP-based exfil if the container has egress).
- Argument injection is also possible when input feeds command **flags**
  rather than a shell string directly — a leading `-` can inject an
  unexpected option into the underlying binary.

### 3. Get the flag
Once you have command execution: `ls` the working directory and common
locations (`/`, `/tmp`, `/app`) for a flag file, then `cat` it directly.

## Pitfall
Don't assume a filtered/blocked first payload means the class is wrong —
identify exactly which character/keyword triggered the block and adjust,
the same way you would for XSS filters.
