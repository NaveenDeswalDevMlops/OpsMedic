# SOP — Internal Project (project tooling, repos, environments, CI/CD, test data)

## Triage checklist
1. Identify the project, tool (repo, board, CI, environment), and whether the issue blocks a release.
2. Check the tool's status page/known-issues channel before individual debugging.
3. Confirm the requester's project role — many requests are access-matrix items in disguise.

## Standard resolution steps
1. Repo/board access: add to the project group after project-lead approval; verify least-privilege role.
2. CI/CD failures: pull the failing job log; distinguish infra failure (runner, quota, secrets) from code failure; requeue after infra fix — code failures go back to the dev team.
3. Environment issues: verify service health, restart the affected pod/service per runbook, validate with a smoke test.
4. Test data requests: provide masked/synthetic data only; production copies require data-owner approval.

## Escalation
Escalate to Platform/DevOps team for runner capacity, secrets rotation, or environment rebuilds.
