---
name: ssrf-deserialization-and-protocol
description: Use for server-side request forgery (a URL/redirect parameter), insecure deserialization (a serialized cookie/object), GraphQL APIs, HTTP request smuggling, unusual HTTP verbs, or unexplained information disclosure.
---

# SSRF, deserialization, GraphQL, and protocol-level attacks

## SSRF
1. Look for a leaked internal hostname or URL first — page source,
   comments, or error text sometimes directly names an internal service
   (e.g. `http://internal-service`, `http://flask_api:8081`).
2. Find a parameter that causes a server-side fetch: `url=`,
   `profile_picture_url=`, a "convert"/"preview"/"webhook" feature,
   a redirect endpoint.
3. Point it at `http://127.0.0.1:<port>/...` or the leaked internal
   hostname. Internal-only endpoints explicitly documented as
   "localhost only" (password reset, admin actions) are exactly the
   target — the request looks local to the server, bypassing that check.
4. **Try a plain `X-Forwarded-For: 127.0.0.1` header spoof first** — some
   apps trust this header directly for IP-based access checks with no
   SSRF or smuggling required at all; only escalate to a full SSRF/
   smuggling chain if the simple header spoof doesn't work.
5. If the target spans two cooperating internal services (e.g. one
   service exposes a "set config" API, another fetches and executes
   whatever that config points to), you may need to chain a write to
   service A with a trigger on service B.

## Insecure deserialization
1. Look for a cookie or parameter holding structured serialized data:
   PHP serialized format (`a:2:{...}`, `O:8:"ClassName"...`), Python
   pickle (often base64, starts with `\x80\x04` or similar opcode bytes
   once decoded), or YAML (`!!python/object:...` / `!!python/object/apply:`
   tags).
2. **PHP**: if the app uses loose comparison (`==`) on a deserialized
   field against a secret, setting that field to the literal boolean
   `true` (serialized as `b:1;`) satisfies `== "any non-empty string"`
   due to PHP's type juggling — this is a common auth-bypass pattern here.
3. **Python pickle/YAML**: if you can control the object being
   deserialized, a payload like
   `!!python/object/apply:subprocess.check_output [["command","args"]]`
   (YAML) or an equivalent pickle `__reduce__` payload achieves RCE
   directly when the deserializer is unsafe (`yaml.load` without
   `SafeLoader`, or a bare `pickle.loads` on user input).
4. If direct command execution seems blocked, try instead injecting an
   object whose attribute is `eval("...")` — if that attribute later gets
   rendered/displayed by the app (e.g. shown in a list), the eval'd
   output (like `os.environ`) leaks through the render, even without a
   direct RCE response.

## GraphQL
1. Check whether introspection is enabled: query `__schema { types { name
   fields { name } } }` — this often reveals fields the frontend never
   uses, including a `flag` field directly on a domain type.
2. Check authorization on every query/mutation argument that takes an ID
   (`userId`, etc.) — GraphQL APIs here frequently skip the "does this ID
   belong to the caller" check entirely (same idea as REST IDOR).
3. A raw-JSON "search"/"filter" argument is a NoSQL injection vector too
   — see `sqli-and-nosqli`.

## HTTP request smuggling (CL.TE / TE.CL)
1. Relevant when there's a proxy chain (e.g. mitmproxy/haproxy/nginx in
   front of the app) and IP-based or session-based access controls you
   can't otherwise reach.
2. Send a request with both `Content-Length` and `Transfer-Encoding:
   chunked` headers, crafted so the frontend and backend disagree about
   where the request ends — this can smuggle a second, hidden request
   that the backend processes as if it came from the proxy itself
   (e.g. as `127.0.0.1`), bypassing IP-based restrictions.
3. This is a heavier technique — try the plain `X-Forwarded-For` spoof
   (see SSRF section above) first; only build a full smuggling PoC if
   that fails.

## Unusual HTTP verbs
- If a path returns 401/403 for GET/POST, try `OPTIONS`, `HEAD`, `PUT`,
  `TRACE` — auth middleware sometimes only guards a subset of methods,
  and `OPTIONS` in particular has bypassed auth entirely in challenges
  here.

## General information disclosure
- `/server-status`, `/.git`, `/robots.txt`, `/icons/README`, and similar
  default/debug paths on Apache/nginx installs can leak internals even
  when the main app is otherwise solid — check these opportunistically.

## Pitfall
These techniques often chain: an SSRF might reach an internal endpoint
that itself needs a deserialization or auth-bypass payload to do anything
useful. Don't stop investigating once you reach the internal service —
check what it actually needs from you.
