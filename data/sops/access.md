# SOP — Access (logins, passwords, MFA, account lockouts, permissions)

## Triage checklist
1. Verify the requester's identity per security policy (employee ID + manager or registered device).
2. Identify the exact system/application and the error shown (locked, expired, 403/denied, MFA failure).
3. Check for a wider outage on the identity provider (AD/SSO) before treating as single-user.

## Standard resolution steps
1. Account locked: unlock in the identity console; reset password with change-at-next-logon enforced.
2. Permission denied: compare the user's group membership against the application access matrix; add the approved entitlement group only if a completed approval exists.
3. MFA issues: remove the stale device registration, guide re-enrolment on the new device, test one SSO login.
4. Confirm access with the user before closing; document the root cause (lockout, missing group, stale MFA).

## Escalation
Escalate to Identity/IAM L2 if: directory replication errors, repeated lockouts within 24h (possible credential stuffing), or entitlement group missing from the matrix.
