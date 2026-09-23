---
name: xss
description: Use when user input (a query param, form field, or search value) is reflected back into the HTML/JS response — the single most common vulnerability class in this benchmark set.
---

# Cross-site scripting (XSS)

Almost every XSS challenge here requires triggering a *specific* string,
often literally `alert('XSS')` or `alert("XSS")` (case-sensitive) — check
the challenge description for the exact required string before assuming
any working XSS is sufficient.

## Method: a systematic filter-bypass ladder

Don't guess randomly. Send a battery of small probes to map exactly what's
blocked, then combine only what's allowed:

1. **Baseline**: does the raw input reflect unescaped at all? Send a
   unique marker string and check if it appears verbatim (not HTML-entity
   encoded) in the response.
2. **What context is it in?** — HTML body, an HTML attribute value
   (`<input value="...">`), or inside a `<script>` block (JS string
   context)? This determines which payload shape applies.
3. **Probe individual blocked elements** one at a time: `<script>`,
   `onerror`, `onload`, `<`, `>`, single quote `'`, double quote `"`,
   space, and specific letters the challenge might filter (`S`, `e`,
   `a`, etc., seen in some challenges here). Note exactly which are
   rejected/stripped vs which pass through.
4. **Pick the smallest working combination** from what survives:
   - `<script>` blocked but `<svg>`/`<img>`/`<details>` allowed:
     `<svg onload=alert('XSS')>`, `<img src=x onerror=alert('XSS')>`,
     `<details ontoggle=alert('XSS') open>`
   - Space blocked/filtered but `/` allowed as separator:
     `<svg/onload=alert('XSS')>`
   - `>` blocked but `<` allowed: browsers tolerate unclosed tags —
     `<img src=x onerror=alert('XSS')` (no closing `>`) can still fire.
   - Single quote blocked, double quote allowed (or vice versa): swap
     quote style in the payload.
   - Attribute-context breakout: if input lands inside
     `value="..."`, close the attribute/tag first:
     `"><img src=x onerror=alert('XSS')>` or, if `>` is blocked,
     `" onmouseover=alert('XSS') x="`.
   - JS-string context (`<script>var x = "INPUT";</script>`) with quotes
     filtered: break out via backslash if unescaped
     (`\"-alert('XSS')//`), or build the required letters at runtime if
     specific letters are blocked: `String.fromCharCode(88,83,83)` or
     `unescape(/%58%53%53/.source)` to construct `"XSS"` without typing
     the filtered letters directly, or `eval('al'+'ert(\"XSS\")')` to
     avoid the literal substring `alert(`.
   - If **all** alphanumeric characters are blocked in a JS context, fall
     back to JSFuck (an encoding using only `[]()!+`) to construct and
     execute arbitrary JS.

## Confirming success without a real browser
You don't have a JS engine to "watch" the alert fire. Treat success as:
the payload is reflected unescaped in a syntactically valid position for
its context, matching exactly what the challenge asks for. Some
challenges also directly return the flag in the HTTP response once the
server-side check recognizes a valid payload was submitted — check the
response body/headers after submitting.

## Pitfall
Don't give up on XSS after one blocked tag — the intended solution is
almost always a *specific* narrow bypass of *one* blocked element, not a
different vulnerability class.
