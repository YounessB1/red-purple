# PoC_basics Experiment Expectation

## Purpose

`PoC_basics` is meant to test whether prompt updates that improve **general pentesting behavior** help the CTF agent solve benchmarks without teaching exploit-specific tutorials.

The behaviors we want GEPA to strengthen are:

- do enough reconnaissance before committing to an exploit path
- identify the most likely sink or control point
- preserve session and cookie state
- validate cheaply before escalating
- revisit privileged or extraction endpoints immediately after a positive signal
- avoid drifting into unrelated bug classes once a plausible path is found

This is **not** a split for teaching:

- specific XSS payload golf
- parser-specific command injection tricks
- benchmark-specific exploit strings

## How GEPA will batch this split

Relevant facts:

- split: [PoC_basics.json](/home/smartdatayb/Desktop/red-purple/source/dataset/PoC_basics.json)
- train size: `24`
- validation size: `5`
- minibatch size: `3`
- GEPA sampler: `EpochShuffledBatchSampler`
- seed: `0`

The train loader preserves the `train` order from the JSON file, and GEPA shuffles that deterministic list with `random.Random(0)`.

Because `24` is divisible by `3`, there is:

- no padding
- exactly `8` iterations per epoch

Train order as stored in the split:

1. `XBEN-001-24`
2. `XBEN-002-24`
3. `XBEN-003-24`
4. `XBEN-020-24`
5. `XBEN-021-24`
6. `XBEN-024-24`
7. `XBEN-027-24`
8. `XBEN-037-24`
9. `XBEN-039-24`
10. `XBEN-043-24`
11. `XBEN-047-24`
12. `XBEN-051-24`
13. `XBEN-052-24`
14. `XBEN-055-24`
15. `XBEN-058-24`
16. `XBEN-060-24`
17. `XBEN-064-24`
18. `XBEN-065-24`
19. `XBEN-069-24`
20. `XBEN-070-24`
21. `XBEN-071-24`
22. `XBEN-072-24`
23. `XBEN-073-24`
24. `XBEN-099-24`

## Exact deterministic order

### Epoch 1

1. `['XBEN-058-24', 'XBEN-001-24', 'XBEN-065-24']`
2. `['XBEN-037-24', 'XBEN-027-24', 'XBEN-051-24']`
3. `['XBEN-073-24', 'XBEN-072-24', 'XBEN-047-24']`
4. `['XBEN-021-24', 'XBEN-003-24', 'XBEN-071-24']`
5. `['XBEN-020-24', 'XBEN-070-24', 'XBEN-024-24']`
6. `['XBEN-069-24', 'XBEN-043-24', 'XBEN-099-24']`
7. `['XBEN-060-24', 'XBEN-064-24', 'XBEN-039-24']`
8. `['XBEN-002-24', 'XBEN-055-24', 'XBEN-052-24']`

### Epoch 2

1. `['XBEN-064-24', 'XBEN-002-24', 'XBEN-073-24']`
2. `['XBEN-058-24', 'XBEN-027-24', 'XBEN-052-24']`
3. `['XBEN-001-24', 'XBEN-021-24', 'XBEN-070-24']`
4. `['XBEN-037-24', 'XBEN-039-24', 'XBEN-069-24']`
5. `['XBEN-071-24', 'XBEN-043-24', 'XBEN-024-24']`
6. `['XBEN-055-24', 'XBEN-051-24', 'XBEN-020-24']`
7. `['XBEN-065-24', 'XBEN-060-24', 'XBEN-047-24']`
8. `['XBEN-072-24', 'XBEN-003-24', 'XBEN-099-24']`

## Validation set

Held-out validation benchmarks:

- `XBEN-012-24` from `xss_reflected_basic`
- `XBEN-033-24` from `ssrf_internal_service_access`
- `XBEN-089-24` from `business_logic_role_escalation`
- `XBEN-090-24` from `idor_object_reference`
- `XBEN-103-24` from `idor_context_override`

What val is meant to show:

- `XBEN-012-24`: can the agent confirm reflection, keep the payload minimal, and follow exact success criteria?
- `XBEN-033-24`: can the agent recognize a fetcher/SSRF surface and pivot inward methodically?
- `XBEN-089-24`: can the agent inspect authenticated state transitions and role-changing opportunities?
- `XBEN-090-24`: can the agent log in, enumerate neighboring objects, and check unauthorized resources systematically?
- `XBEN-103-24`: can the agent inspect cookies/headers and discover context override instead of only obvious query or form parameters?

## Iteration-by-iteration expectation

## Iteration 1

Batch:

- `XBEN-058-24` (`idor_object_reference`)
- `XBEN-001-24` (`idor_object_reference`)
- `XBEN-065-24` (`xss_reflected_basic`)

What this batch is teaching:

- start with lightweight recon on the authenticated surface
- when IDs or object references appear, try adjacent objects before inventing a new exploit class
- when reflected input appears, confirm it with a marker and use the smallest valid proof payload

Challenge-specific learning:

- `XBEN-058-24`: IDOR GET case that rewards noticing the direct object reference quickly and trying adjacent values
- `XBEN-001-24`: authenticated object-reference case that reinforces login, object enumeration, and immediate extraction checking
- `XBEN-065-24`: basic reflected XSS case that reinforces “find sink -> validate -> minimal payload”

Expected prompt shift:

- “After login, inspect object-bearing routes before escalating.”
- “After confirming reflection, do not overcomplicate the payload.”

## Iteration 2

Batch:

- `XBEN-037-24` (`cmdi_ping_endpoint_support`)
- `XBEN-027-24` (`idor_context_override`)
- `XBEN-051-24` (`xss_reflected_basic`)

What this batch is teaching:

- read endpoint purpose before picking a payload
- inspect request state holistically: parameters, headers, cookies, and any user-context selectors
- once a plausible sink is found, keep pressure on that sink instead of drifting

Challenge-specific learning:

- `XBEN-037-24`: ping-style command injection support case; teaches “recognize execution sink from endpoint semantics”
- `XBEN-027-24`: reinforces that user context may be selected indirectly, not only through obvious route parameters
- `XBEN-051-24`: basic reflected XSS again, reinforcing minimal proof and anti-drift

Expected prompt shift:

- “Infer the likely primitive from the route’s purpose.”
- “Inspect request metadata before abandoning the current hypothesis.”

## Iteration 3

Batch:

- `XBEN-073-24` (`cmdi_ping_endpoint_support`)
- `XBEN-072-24` (`business_logic_role_escalation`)
- `XBEN-047-24` (`xss_reflected_basic`)

What this batch is teaching:

- confirm exploitability with the cheapest possible probe
- after any state change, re-check privileged pages immediately
- do not mistake “I know the bug class” for “I have extracted the flag”

Challenge-specific learning:

- `XBEN-073-24`: another ping-style command-injection support case; encourages simple command-separator validation and extraction focus
- `XBEN-072-24`: business-logic privilege escalation; teaches checking role or state transitions right after login
- `XBEN-047-24`: reflected XSS baseline; reinforces minimal success criteria

Expected prompt shift:

- “Probe cheaply first; escalate second.”
- “Whenever privilege seems to change, revisit admin-only routes immediately.”

## Iteration 4

Batch:

- `XBEN-021-24` (`idor_context_override`)
- `XBEN-003-24` (`idor_context_override`)
- `XBEN-071-24` (`sqli_auth_bypass_support`)

What this batch is teaching:

- inspect ambient state and request metadata before assuming the bug is elsewhere
- treat login and identity-selection surfaces as high-value early probes
- once authenticated or bypassed, pivot directly to privileged views and stored data

Challenge-specific learning:

- `XBEN-021-24`: cookie/context override style IDOR
- `XBEN-003-24`: header-based context override
- `XBEN-071-24`: SQLi auth-bypass support case that reinforces the prompt habit of checking auth flows early

Expected prompt shift:

- “Check whether user identity is chosen by client-controlled state.”
- “Auth flows are early targets, not a late-stage fallback.”

## Iteration 5

Batch:

- `XBEN-020-24` (`ssrf_internal_service_access`)
- `XBEN-070-24` (`xss_reflected_basic`)
- `XBEN-024-24` (`ssrf_internal_service_access`)

What this batch is teaching:

- during recon, identify URL fetchers and outbound request surfaces
- once a fetch primitive is confirmed, try loopback and internal targets methodically
- do not let a familiar bug class like XSS distract from a more structurally promising fetcher primitive

Challenge-specific learning:

- `XBEN-020-24`: SSRF support case; teaches recognizing fetcher behavior
- `XBEN-070-24`: another minimal XSS case that keeps the prompt anchored in simple validation
- `XBEN-024-24`: second SSRF support case that reinforces internal pivoting

Expected prompt shift:

- “A user-controlled fetcher is a first-class exploit surface.”
- “After the first internal response, enumerate nearby internal routes before switching tactics.”

## Iteration 6

Batch:

- `XBEN-069-24` (`idor_object_reference`)
- `XBEN-043-24` (`idor_object_reference`)
- `XBEN-099-24` (`idor_object_reference`)

What this batch is teaching:

- a very concentrated lesson in authenticated object enumeration
- compare legitimate resources to likely neighboring unauthorized ones
- after an unauthorized object opens, inspect it fully for flag-bearing subresources

Challenge-specific learning:

- all three are direct object-reference cases
- this batch should strongly reward better enumeration discipline and lower the temptation to drift into irrelevant exploit classes

Expected prompt shift:

- “When a benchmark exposes object IDs, enumeration is not optional.”
- “Object-reference exploitation should be exhausted before trying unrelated attacks.”

## Iteration 7

Batch:

- `XBEN-060-24` (`business_logic_role_escalation`)
- `XBEN-064-24` (`xss_reflected_basic`)
- `XBEN-039-24` (`sqli_auth_bypass_support`)

What this batch is teaching:

- probe role/state transitions and auth surfaces early
- treat privilege change as a state machine problem, not necessarily as an injection problem
- once login or privilege escalation succeeds, check admin-only endpoints immediately

Challenge-specific learning:

- `XBEN-060-24`: business-logic privilege case, good for reinforcing authenticated state analysis
- `XBEN-064-24`: basic XSS, again reinforcing minimalism and exact proof discipline
- `XBEN-039-24`: SQLi auth-bypass support case, reinforcing “check login first”

Expected prompt shift:

- “Privilege escalation often begins with ordinary authenticated features.”
- “Successful auth transitions should be exploited immediately, not admired.”

## Iteration 8

Batch:

- `XBEN-002-24` (`idor_object_reference`)
- `XBEN-055-24` (`business_logic_role_escalation`)
- `XBEN-052-24` (`business_logic_role_escalation`)

What this batch is teaching:

- combine two of the split’s most important generic behaviors:
  - enumerate reachable state/objects
  - inspect low-privilege authenticated features for role changes
- this is a very “prompt basics” batch: no exotic payloads, mostly disciplined reasoning and orderly follow-through

Challenge-specific learning:

- `XBEN-002-24`: reinforces adjacent-object exploration
- `XBEN-055-24` and `XBEN-052-24`: reinforce role/state abuse and immediate revisit of protected pages

Expected prompt shift:

- “Most wins come from exploiting ordinary app state carefully.”
- “The next action after a state change is to revisit privileged pages, not to keep brainstorming.”

## If the run continues past 8 iterations

At iteration `9`, GEPA starts epoch `2` and reshuffles deterministically again.

The next two batches are:

9. `['XBEN-064-24', 'XBEN-002-24', 'XBEN-073-24']`
10. `['XBEN-058-24', 'XBEN-027-24', 'XBEN-052-24']`

Interpretation:

- epoch `2` revisits the same archetypes in new combinations
- if the prompt improvements are truly general, they should keep helping even when XSS, IDOR, CMDI support, and role-escalation cases are mixed differently

## What success should look like

The desired improvement pattern is:

- train improvements on several different vulnerability classes
- especially on batches mixing IDOR, XSS, SSRF, business logic, and auth support cases
- followed by better validation on:
  - `XBEN-090-24`
  - `XBEN-103-24`
  - `XBEN-012-24`
  - `XBEN-089-24`
  - `XBEN-033-24`

The strongest evidence for the PoC would be:

- the prompt becomes better at recon and sink selection
- the agent spends less time on wrong bug classes
- it preserves session state more reliably
- it validates early and cheaply
- it extracts immediately after a positive signal

That would support the claim that GEPA is improving the **general pentesting loop**, not merely learning exploit-specific micro-tutorials.
