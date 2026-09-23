---
name: jwt-crypto-and-cve
description: Use when the app uses a JWT, when you spot loose-comparison/type-juggling-prone code (PHP), when session cookies look encrypted, or when a fingerprinted software/version suggests a known CVE.
---

# JWT, crypto/type-juggling quirks, and known CVEs

## JWT attacks
1. Decode the JWT (base64) and inspect header + payload — look for role/
   privilege/user-identity claims (`role`, `is_admin`, `username`,
   `user_id`).
2. **`alg: none` forgery**: change the header to `{"alg":"none","typ":
   "JWT"}`, edit the payload claims you want (e.g. `role: admin`,
   `username: admin`), and submit the token as `header.payload.` (base64
   header, base64 payload, empty signature — trailing dot kept). Many
   JWT libraries still accept this.
3. Watch for a **split trust** pattern: some apps validate a *separate*
   session cookie for authenticity but then trust the *JWT's* claims for
   identity/role. If so, you may need a genuinely valid session cookie
   from a login you control, paired with a forged JWT claiming a
   different (higher-privilege) identity — sent together.
4. If `none` doesn't work, also consider algorithm confusion (RS256 →
   HS256 using the public key as the HMAC secret) if a public key is
   discoverable.

## PHP type-juggling / "magic hash" quirks
- PHP's loose comparison (`==`) treats certain strings as numbers. A
  hash string shaped like `"0e" + all digits` (e.g.
  `0e462097431906509019562988736854`) is interpreted as `0 × 10^…` = `0`,
  so **any two such "magic hash" strings compare equal with `==`**. If you
  know a target password hash starts with `0e` followed only by digits,
  search for (or recall) an input string with a matching magic-hash
  digest and submit that as the password.
- `strcmp($input, $secret)` (and similar comparison functions) return
  `NULL` when given an **array** instead of a string in older PHP — and
  `NULL == false`/`0` in loose comparison. Submitting a parameter as an
  array (e.g. `password[]=x` in a form/query string) can bypass a
  string-comparison password check entirely.

## Cookie/session crypto weaknesses
- **AES-CBC without a MAC**: if a session cookie is AES-CBC encrypted with
  no integrity check, flipping a byte in one ciphertext block predictably
  corrupts the *next* block's plaintext after decryption (bit-flipping
  attack) — useful for changing a specific field (e.g. a username or
  role) in an otherwise-opaque encrypted cookie, if you know/can guess
  its position.
- **Padding oracle**: if the server returns a *distinguishable* error for
  "invalid padding" vs other failures (different status code, different
  message) when given a tampered ciphertext, you can decrypt (and forge)
  the cookie byte-by-byte without knowing the key, using a standard
  padding-oracle attack. This is mechanical but slow — script it rather
  than doing it by hand.

## Known-CVE workflow
When recon fingerprints specific software + a version number (WordPress
plugin, Apache, Struts, etc.), treat "check for a known CVE" as a
first-class step, not a last resort:
1. Note the exact product + version string.
2. Recall/search for public CVEs affecting that version — this benchmark
   set includes real CVEs (e.g. Apache 2.4.49/2.4.50 mod_cgi RCE
   CVE-2021-41773/CVE-2021-42013 — see `file-path-and-upload`; WordPress
   plugin RCEs reachable via `php://filter`/`data://` wrapper chains;
   Struts2 `.action` endpoints suggesting OGNL injection).
3. Apply the known exploit technique for that CVE directly rather than
   rediscovering the bug class from scratch.

## Pitfall
Don't assume a JWT with a valid-looking signature is unforgeable without
testing `alg: none` and checking for a split session/JWT trust pattern
first — both are cheap to test and recur in this benchmark set.
