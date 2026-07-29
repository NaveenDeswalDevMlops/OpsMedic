# SOP — IT Support queue

Scope: internal IT and infrastructure issues (endpoints, office network, corporate apps, servers).

## Triage
1. Identify asset/user, affected service, and blast radius (single user vs many).
2. Check monitoring/known outages first; link to the parent incident if one exists.

## Standard resolution flow
1. Single user: standard endpoint fixes (restart service, re-join network, reinstall client, permissions check).
2. Multiple users: treat as incident — engage the owning infra team, post a status note, track to restore.
3. Confirm restoration with the reporter and record root cause.

## Escalation
Escalate to Infrastructure/Network L2 for server-side faults, capacity issues, or anything needing privileged changes.
