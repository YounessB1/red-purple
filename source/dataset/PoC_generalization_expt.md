# PoC Generalization Expectations

This note describes what we expect to happen in the first `10` GEPA training iterations for the current [PoC_generalization.json](/home/smartdatayb/Desktop/red-purple/source/dataset/PoC_generalization.json).

Assumptions used here:

- split: `PoC_generalization`
- train minibatch size: `3`
- GEPA seed: `0`
- train order is deterministic
- one GEPA training iteration consumes one train minibatch

Current validation set:

- `XBEN-004-24` (`xss_event_handler_bypass`)
- `XBEN-018-24` (`xss_filter_bypass`)
- `XBEN-022-24` (`ssti_python_object_traversal`)
- `XBEN-045-24` (`cmdi_ping_family`)
- `XBEN-090-24` (`idor_object_reference_curated`)

Current train families:

- `ssti_python_object_traversal`: `XBEN-023-24`, `XBEN-053-24`, `XBEN-063-24`, `XBEN-076-24`, `XBEN-080-24`, `XBEN-104-24`
- `xss_event_handler_bypass`: `XBEN-008-24`, `XBEN-011-24`, `XBEN-015-24`, `XBEN-016-24`, `XBEN-046-24`, `XBEN-062-24`
- `xss_filter_bypass`: `XBEN-048-24`, `XBEN-049-24`, `XBEN-050-24`, `XBEN-074-24`
- `idor_object_reference_curated`: `XBEN-001-24`, `XBEN-002-24`, `XBEN-058-24`, `XBEN-069-24`
- `cmdi_ping_family`: `XBEN-037-24`, `XBEN-073-24`

Important caveat:

- GEPA may skip reflection on a minibatch if the sampled train examples already score perfectly.
- So the schedule below tells us what benchmarks are presented each iteration and what kind of learning signal we hope they create, not a guarantee that every iteration will produce a meaningful update.

## Deterministic Train Batches

Epoch 1:

1. `XBEN-049-24`, `XBEN-080-24`, `XBEN-074-24`
2. `XBEN-023-24`, `XBEN-001-24`, `XBEN-076-24`
3. `XBEN-050-24`, `XBEN-008-24`, `XBEN-011-24`
4. `XBEN-048-24`, `XBEN-016-24`, `XBEN-037-24`
5. `XBEN-015-24`, `XBEN-073-24`, `XBEN-062-24`
6. `XBEN-104-24`, `XBEN-063-24`, `XBEN-069-24`
7. `XBEN-046-24`, `XBEN-002-24`, `XBEN-058-24`
8. `XBEN-053-24`

Epoch 2:

9. `XBEN-002-24`, `XBEN-073-24`, `XBEN-058-24`
10. `XBEN-001-24`, `XBEN-076-24`, `XBEN-015-24`

## Iteration 1

Train batch:

- `XBEN-049-24` `xss_filter_bypass`
- `XBEN-080-24` `ssti_python_object_traversal`
- `XBEN-074-24` `xss_filter_bypass`

Expectation:

- This batch is dominated by `xss_filter_bypass`, with one SSTI family member.
- We want GEPA to start learning a reusable XSS filter-analysis loop:
  - confirm reflection
  - infer what the filter blocks
  - mutate payloads systematically instead of randomly
- The SSTI member should keep the update from collapsing into an over-specific XSS payload recipe.

Expected transfer link:

- Better filter inference here should later help `XBEN-018-24` val.

## Iteration 2

Train batch:

- `XBEN-023-24` `ssti_python_object_traversal`
- `XBEN-001-24` `idor_object_reference_curated`
- `XBEN-076-24` `ssti_python_object_traversal`

Expectation:

- This is the first batch where `2/3` examples reinforce the SSTI object-traversal family.
- We want the agent to learn:
  - arithmetic or low-risk template probing first
  - identify Python/Jinja behavior
  - escalate toward object traversal and file read/RCE
- The IDOR example should also reinforce that once authenticated access exists, neighboring objects are worth probing immediately.

Expected transfer link:

- Better SSTI escalation here should later help `XBEN-022-24` val.

## Iteration 3

Train batch:

- `XBEN-050-24` `xss_filter_bypass`
- `XBEN-008-24` `xss_event_handler_bypass`
- `XBEN-011-24` `xss_event_handler_bypass`

Expectation:

- This is a strong XSS batch.
- `2/3` examples are event-handler bypass, while the third is still another filtered XSS case.
- We want the agent to improve:
  - early recognition that classic `<script>` style payloads are blocked
  - selection of alternate tags/events
  - tighter adaptation to observed filter behavior

Expected transfer link:

- This should support both `XBEN-004-24` and `XBEN-018-24` val from one broader XSS decision procedure.

## Iteration 4

Train batch:

- `XBEN-048-24` `xss_filter_bypass`
- `XBEN-016-24` `xss_event_handler_bypass`
- `XBEN-037-24` `cmdi_ping_family`

Expectation:

- This is the first command-injection exposure.
- We want the agent to learn a reusable CMDI loop:
  - identify ping-style functionality
  - try simple separators or timing probes
  - confirm execution before extraction attempts
- The XSS cases should keep the update focused on family selection rather than one endpoint-specific trick.

Expected transfer link:

- Any useful CMDI learning here is groundwork for held-out `XBEN-045-24`.

## Iteration 5

Train batch:

- `XBEN-015-24` `xss_event_handler_bypass`
- `XBEN-073-24` `cmdi_ping_family`
- `XBEN-062-24` `xss_event_handler_bypass`

Expectation:

- `2/3` examples are event-handler bypass, with the second ping-family command injection case in the middle.
- We want:
  - more robust event-trigger selection under filtering constraints
  - better confidence on ping-endpoint exploitation after seeing both CMDI train members

Expected transfer link:

- This iteration is one of the best chances to improve `XBEN-004-24`, while also making `XBEN-045-24` more reachable.

## Iteration 6

Train batch:

- `XBEN-104-24` `ssti_python_object_traversal`
- `XBEN-063-24` `ssti_python_object_traversal`
- `XBEN-069-24` `idor_object_reference_curated`

Expectation:

- This is one of the highest-signal batches in the run.
- `2/3` examples reinforce Python/Jinja SSTI escalation, while the IDOR example reinforces authenticated object enumeration.
- We want the agent to improve:
  - SSTI escalation consistency
  - less wasted motion before reaching builtins / object traversal
  - stronger instinct to enumerate authenticated object references after login

Expected transfer link:

- Strong transfer opportunity for `XBEN-022-24`, and some support for `XBEN-090-24`.

## Iteration 7

Train batch:

- `XBEN-046-24` `xss_event_handler_bypass`
- `XBEN-002-24` `idor_object_reference_curated`
- `XBEN-058-24` `idor_object_reference_curated`

Expectation:

- This is the best dedicated IDOR-support batch in epoch 1.
- `2/3` examples teach authenticated object-reference abuse:
  - identify object-bearing endpoints
  - try nearby IDs
  - verify whether authorization follows the object or only the session
- The XSS member prevents the update from becoming too benchmark-local.

Expected transfer link:

- This is one of the key batches for `XBEN-090-24` transfer.

## Iteration 8

Train batch:

- `XBEN-053-24` `ssti_python_object_traversal`

Expectation:

- This short batch has only one example, but it is a clean family representative.
- We want the agent to sharpen a minimal SSTI playbook:
  - probe with arithmetic
  - confirm template evaluation
  - escalate to Python object traversal

Expected transfer link:

- A successful update here should be highly reusable for `XBEN-022-24`.

## Iteration 9

Train batch:

- `XBEN-002-24` `idor_object_reference_curated`
- `XBEN-073-24` `cmdi_ping_family`
- `XBEN-058-24` `idor_object_reference_curated`

Expectation:

- This is the first batch of epoch 2, so it tests whether learned family-level behavior is stable after reshuffling.
- `2/3` examples are curated IDOR supports, with one CMDI support example.
- We want:
  - stable authenticated ID enumeration behavior
  - stable ping-family command-injection discovery and validation

Expected transfer link:

- This iteration should support both `XBEN-090-24` and `XBEN-045-24`.

## Iteration 10

Train batch:

- `XBEN-001-24` `idor_object_reference_curated`
- `XBEN-076-24` `ssti_python_object_traversal`
- `XBEN-015-24` `xss_event_handler_bypass`

Expectation:

- This is a mixed family batch, so the main gain should be a stronger exploit-family branching policy.
- We want the agent to improve:
  - when to pursue authenticated object enumeration
  - when to classify behavior as SSTI and escalate toward Python traversal
  - when to switch to event-handler bypass under reflected XSS constraints

Expected transfer link:

- This kind of batch helps the global claim: GEPA should not learn one isolated trick per benchmark, but a set of reusable exploit strategies that are selected and reused appropriately.

## What Success Should Look Like

Across these `10` iterations, the PoC should ideally show:

- repeated train improvements across the same family rather than isolated one-off wins
- better exploit-family selection early in each trajectory
- transfer first on the strongest families:
  - `XBEN-022-24` from repeated SSTI object-traversal training
  - `XBEN-004-24` from repeated event-handler bypass training
  - `XBEN-090-24` from curated authenticated IDOR training
- more partial or caveated transfer on:
  - `XBEN-018-24`, because the family documentation is noisier
  - `XBEN-045-24`, because it adds parser-specific output shaping on top of the shared ping-style CMDI technique

The cleanest argument this split aims to support is:

- GEPA does not need to learn one bespoke skill per benchmark.
- It can learn reusable exploit techniques at the family level.
- Those techniques can then transfer to unseen validation benchmarks drawn from the same family.
