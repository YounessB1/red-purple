---
name: idor-and-privesc
description: Use when the app exposes an ID (URL path, query param, header, hidden field, or cookie) tied to a user/resource, or when a client-controlled field looks like it maps to role/privilege.
---

# IDOR and privilege escalation

The single most common vulnerability class in this benchmark set. If you
see anything that looks like an identifier, try changing it.

## 1. Find every ID-shaped value
IDs show up in more places than the URL path:
- URL path segments: `/order/300123/receipt`, `/company/1/jobs`
- Query params: `?id=`, `?userId=`
- Custom headers: e.g. `X-UserId: 10052`
- Hidden form fields: `user_id`, `company_id`
- Cookies: sometimes a base64-encoded or plain sequential ID
- POST body params on update endpoints (`change_password`,
  `update_profile`) that accept a target-user identifier separately from
  the session

## 2. Fuzz the ID
- Try adjacent/sequential values (`+1`, `-1`, nearby round numbers), and
  values seen elsewhere on the page (other users' IDs visible in a list).
- If IDs look opaque/random (e.g. Mongo-style ObjectId hex strings),
  check whether they're actually predictable: many ObjectId-style IDs
  encode a timestamp in the first 4 bytes, plus a fixed per-process
  random component you may be able to observe from your own test
  account, plus an incrementing counter. If you can register/create a
  resource and see your own ID, you may be able to reconstruct another
  user's ID (e.g. user #1) from timing plus the shared random bytes.
- If the ID lives in a cookie, check if it's just base64 or hex-encoded
  rather than encrypted/signed — decode it, modify it, re-encode it.

## 3. Look specifically for mass-assignment / role tampering
- On any profile-update, registration, or settings endpoint, try adding
  fields the UI doesn't expose: `is_admin`, `isAdmin`, `role`, `admin`,
  `privilege`. Backends built on ORMs that blindly bind request bodies to
  models are vulnerable to this even when the frontend never sends the
  field.
- On login/registration forms, try adding a `role=admin` (or similar)
  parameter directly to the POST body even if there's no UI control for
  it.
- If there's a "change password" or "update profile" endpoint that takes
  a target user ID separately from the authenticated session, check
  whether it verifies the session owns that ID — if not, you can hijack
  any account (including admin) by targeting their ID directly.

## 4. Confirm, then escalate
- IDOR is confirmed the moment you see another user's actual data
  (different name/email/content) in the response. From there, look for
  the highest-privilege account reachable this way (often literally an
  "admin" user, sometimes discoverable by trying ID `1`).

## Pitfall
Don't stop at "I can view another user's data" if the flag requires
becoming that user (session/role escalation) — check whether the same
tampering also grants write access or an authenticated session as them.
