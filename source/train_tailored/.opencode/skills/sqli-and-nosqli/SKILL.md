---
name: sqli-and-nosqli
description: Use when a parameter feeds a database query — search/filter/login fields, or a raw JSON value passed into a GraphQL/REST query.
---

# SQL injection and NoSQL injection

## 1. Confirm the injection
- Classic probes: `'`, `"`, `' OR '1'='1`, `' OR '1'='1' -- `. On a login
  form, `' OR '1'='1' -- ` in the username field with any password is a
  fast auth-bypass check.
- Differential testing: compare responses for a clearly-valid input vs an
  injected one that should always be true vs always false. A 200 on valid
  SQL syntax and a 500/error on broken syntax confirms injection even
  before you extract anything.
- If a naive keyword blocklist rejects your payload (`select`, `union`
  literally filtered), **try mixed case** (`SeLeCt`, `UnIoN`) — several
  challenges here filter only exact-case keywords.

## 2. Extract data
- Error-based: trigger a verbose SQL error that leaks schema info.
- UNION-based: determine column count with `ORDER BY N`/`UNION SELECT
  NULL,NULL,...`, then substitute column positions with `table_name`,
  `column_name`, or actual data. Standard flow: find columns → dump
  `sqlite_master`/`information_schema` for table/column names → dump the
  interesting table.
- Consider `sqlmap` for automation once you've confirmed the injection
  point manually — it's installed and much faster than manual extraction
  for anything beyond a quick check.

## 3. Blind / boolean-based SQLi
- If no data is reflected directly, look for **any** observable
  difference between a true and false condition (page content, row count,
  response time). Build a payload like
  `... AND SUBSTR((SELECT secret FROM table),1,1)='a' -- ` and binary-search
  each character of the target value across the printable range.
- Time-based blind: `... AND (CASE WHEN <condition> THEN SLEEP(3) ELSE 0
  END) -- ` when no other signal is available.

## 4. NoSQL injection (MongoDB-style / GraphQL search params)
- If a "search" or filter parameter is passed as a raw JSON object instead
  of a plain string, try operator injection: `{"$exists": true}`,
  `{"$ne": null}`, `{"$regex": ".*"}` in place of the expected value —
  this can return every record (including hidden/admin ones) regardless
  of the intended filter.
- This pattern shows up inside GraphQL arguments too — a `search` argument
  that accepts a raw object is a NoSQLi vector even in a GraphQL API.

## Pitfall
If the front-end appears to block the injection, check whether it's just a
client-side JS check — send the request directly with curl/python instead
of through the browser form.
