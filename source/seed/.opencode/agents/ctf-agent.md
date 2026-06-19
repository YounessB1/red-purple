---
description: "CTF penetration testing agent"
model: "openrouter/qwen/qwen3-32b"
temperature: 0.0
maxSteps: 50
tools:
  bash: true
  read: true
  write: true
  edit: true
permission:
  bash: "allow"
  read: "allow"
  write: "allow"
  edit: "allow"
  glob: "allow"
  grep: "allow"
  skill: "allow"
  external_directory: "allow"
  doom_loop: "allow"
  webfetch: "deny"
  websearch: "deny"
  question: "deny"
  lsp: "deny"
  task: "deny"
---
