---
name: xss-context-reflection
description: >
  Use when tackling reflected or stored XSS where input is echoed into HTML.
  Triggers: mentions of XSS, reflected parameters, filters blocking <script>, quotes, or parentheses;
  evidence that input appears after a closed tag or inside attributes/text. Not for CSP bypass specifics or DOM-only gadgets.
---

Context-first XSS exploitation

Why this matters
- XSS success depends on the injection context. Guessing payloads without knowing where your bytes land wastes time and misses easy wins.

Identify the reflection context
- Inside an attribute value (quoted or unquoted)
- Inside element text (between tags)
- Inside a JavaScript context (<script> ... )
- After a closed tag (e.g., reflected right after </input> or >)

General rules
- Produce syntactically valid HTML/JS for the exact context.
- Prefer auto-firing payloads that don’t need user interaction.
- If filters block keywords or symbols, try minimal obfuscation first (case, backticks, benign events), then encoding.

Cookbook by context
1) After a closed tag (very common in CTF forms)
   - You must create a new element; bare attributes won’t bind.
   - Minimal auto-fire examples:
     - <input autofocus onfocus=alert`1`>
     - <svg onload=alert`1`></svg>
     - <img src=x onerror=alert`1`>
   - Rationale: autofocus triggers focus automatically; onfocus fires without clicks.

2) Inside an unquoted attribute value
   - Break out with a space to start a new attribute:
     - value=abc onmouseover=alert`1`
   - If space is filtered, try a forward slash or newline if tolerated.

3) Inside a quoted attribute value
   - Close the quote, add a space, then your event, then ensure the tag remains valid:
     - value="abc" autofocus onfocus=alert`1`
   - If quotes are blocked, try switching to the other quote type or to unquoted contexts via entity decoding.

4) Inside element text (between tags)
   - Inject a closing tag and a new element:
     - </p><svg onload=alert`1`>

5) Inside <script> … </script>
   - Use JS that survives filters; alert`1` avoids parentheses/quotes if blocked.
   - Fallbacks: setTimeout`alert\`1\``, print(), confirm`1` if alert is filtered.

Filter probing and bypass tips
- Case sensitivity: test OnFoCuS vs onfocus.
- Reserved word filters: avoid the literal string "script"; prefer event handlers or SVG/IMG vectors.
- Symbol filters: If quotes/parentheses are blocked, use template literals: alert`XSS`.
- Encodings: Try HTML entity encoding for symbols (&gt;, &quot;), or partial URL-encoding for filtered bytes.
- Attribute-less vectors: <svg/onload=alert`1`> works in many parsers (no closing > needed in some cases), but validate if parser is strict.

Validation heuristics
- Confirm that your input is reflected unchanged and where. If the server echoes “Sorry you can’t use: 'script'”, it’s a filter hint, not evidence of client JS execution—stay context-driven.
- If you see your payload right after a > or </tag>, switch to the "after a closed tag" technique.

Common pitfalls (and what they indicate)
- Trying bare attributes when not inside a tag ⇒ you’re after a closed tag; create a new element.
- Fixating on server error messages unrelated to the browser context ⇒ re-check reflection location.
- Only trying <script> … ⇒ likely blacklisted; switch to events or SVG/IMG.

Minimal workflow
1) Fetch the page with your marker (e.g., AAA) and view source to locate reflection.
2) Choose the appropriate context pattern from the cookbook.
3) Test with quote/paren-free alert`1` using an autotrigger event.
4) If blocked, probe case sensitivity and try a different safe event.
5) Once code execution is proven, pivot to retrieving the flag if it’s revealed via XSS (cookies/admin actions) per challenge specifics.
