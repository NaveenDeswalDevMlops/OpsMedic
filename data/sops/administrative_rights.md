# SOP — Administrative rights (local admin, elevated privileges, install rights)

## Triage checklist
1. Confirm what the user is trying to do that requires elevation (software install, driver, settings).
2. Check whether a packaged/self-service install exists in the software portal (preferred over granting admin).
3. Verify the request has manager + application-owner approval where policy requires it.

## Standard resolution steps
1. Prefer the software portal package or a remote-assisted install by the service desk.
2. If elevation is justified, grant TIME-BOUND local admin via the endpoint privilege tool (e.g., 24h), never permanent.
3. Record the grant with ticket reference, scope, and expiry; verify auto-revocation fired.
4. For recurring needs, route the user to the standing-exception process with security sign-off.

## Escalation
Escalate to Endpoint Security if: the request involves servers, security tooling changes, or disabling protection agents.
