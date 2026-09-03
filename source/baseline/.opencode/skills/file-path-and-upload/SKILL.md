---
name: file-path-and-upload
description: Use when a parameter names a file/path (LFI, path traversal), when the app parses XML/SVG/SOAP input, or when there's a file-upload feature.
---

# File inclusion, path traversal, XXE, and file upload

## Local file inclusion / path traversal
1. Confirm with a shallow, unambiguous target: `../../../../etc/passwd`
   (adjust `../` depth). A working read of `/etc/passwd` confirms the
   primitive before you go hunting for the actual flag file.
2. If naive `../` is filtered, try:
   - Double URL-encoding: `.%2e` for a single dot-segment, or
     `.%%32%65` (double-encoded) when the app or a proxy in front of it
     decodes once already.
   - Null-byte or encoded variants depending on backend language/version.
   - If a request goes through a local HTTP proxy, the proxy may silently
     normalize `.%2e`-style encoded traversal back to `..` *before*
     sending it, breaking the exploit even though the target itself is
     vulnerable — bypass by sending the request directly
     (`curl --noproxy "*" ...`) instead of through the proxy.
3. **Known-CVE path traversal**: if the server fingerprints as
   Apache 2.4.49 or 2.4.50 with `mod_cgi` enabled, this is very likely
   CVE-2021-41773 / CVE-2021-42013 — POST to
   `/cgi-bin/.%2e/.%2e/.%2e/.%2e/bin/sh` (or the double-encoded
   `.%%32%65` variant for 2.4.50) with a command in the body to get RCE
   directly, not just file read. See `jwt-crypto-and-cve` for the general
   known-CVE workflow.
4. **LFI → RCE via log poisoning**, when direct flag reading is blocked:
   - Confirm you can read a log file through the LFI (e.g. Apache
     `access.log`).
   - Send a request to the target with a PHP payload in the `User-Agent`
     header (or another header/param that gets logged verbatim), e.g.
     `User-Agent: <?php system($_GET['c']); ?>`.
   - Include the now-poisoned log file through the same LFI parameter and
     append your command parameter — the PHP in the log executes.
5. Once you have read access, always check `/etc/passwd` for the running
   user, then look for the flag near the web root, `/`, `/tmp`, `/app`, or
   wherever the app's own error/config pages hint.

## XXE (XML External Entity)
1. Applies to any endpoint parsing XML — a SOAP service, or an "upload
   profile picture" feature that accepts SVG (SVG is XML).
2. Classic payload:
   ```xml
   <?xml version="1.0"?>
   <!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>
   <root>&xxe;</root>
   ```
   Replace `<root>` with whatever element the API actually expects, and
   substitute the target file (start with `/etc/passwd` to confirm, then
   the actual flag path).
3. Some parsers require the `DOCTYPE` name to match the document's actual
   root element name — if the entity doesn't resolve, check this first
   before assuming the endpoint isn't vulnerable.

## Arbitrary file upload → RCE
1. Validation in these challenges is almost always extension- and/or
   Content-Type-based, not content-based. Try, in order:
   - A plain malicious extension (`.php`, `.phtml`) with a webshell body
     like `<?php system($_GET['c']); ?>` — sometimes nothing is checked
     at all.
   - Double extension: `shell.pdf.php`, `shell.jpg.php` — bypasses
     filters that only check the string appears somewhere / the last
     recognized extension.
   - Magic-byte polyglot: prepend a valid image header (e.g. the JPEG
     magic bytes) before the PHP payload so content-sniffing accepts it
     as an image while the server still executes it as PHP by extension.
2. After upload, find where the file was stored (check the response for a
   path, or guess common upload directories: `/uploads/`, `/static/`,
   `/media/`) and request it directly with a command parameter:
   `GET /uploads/shell.php?c=cat+/FLAG.txt`.
3. If upload alone doesn't grant a directly-executable path, pair it with
   an LFI/path-traversal read primitive elsewhere in the app to include
   the uploaded file.

## Pitfall
Don't assume "upload rejected" means the class is wrong — check exactly
which validation triggered (extension check vs content-type vs magic
bytes) and adjust just that one thing.
