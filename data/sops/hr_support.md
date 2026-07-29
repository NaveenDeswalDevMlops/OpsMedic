# SOP — HR Support (onboarding/offboarding, payroll/leave portal, HR systems)

## Triage checklist
1. Classify: new-hire setup, leaver deactivation, HR portal error, or payroll/leave data issue.
2. Confirm effective dates from the HR system record — many "issues" are pending start dates.
3. For data changes, verify the request comes from the employee themselves or authorized HR staff.

## Standard resolution steps
1. Onboarding: confirm the joiner record synced from HRMS; trigger account provisioning workflow; verify email, SSO, and mandatory apps.
2. Offboarding: on the leaver date, disable accounts, revoke tokens/sessions, set mailbox delegation per manager request, reclaim assets.
3. Portal errors: reproduce, clear browser cache/SSO session, test in another browser; if server-side, raise to the HRMS application team with screenshots.
4. Payroll/leave data: never edit directly; route to HR operations with the ticket reference.

## Escalation
Escalate to HRMS application support for workflow/sync failures; to IAM for provisioning stuck >4h.
