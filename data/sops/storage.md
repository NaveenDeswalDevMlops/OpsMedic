# SOP — Storage (shared drives, quotas, backups/restores, cloud storage)

## Triage checklist
1. Identify the storage type (network share, cloud drive, mailbox archive) and exact path/URL.
2. Classify: access issue, quota full, missing/deleted data, or performance.
3. For deletions, capture when the data was last seen and the retention window applicable.

## Standard resolution steps
1. Access: verify group membership on the share ACL; add via approved request only; test with the user.
2. Quota full: identify large/old data with the user, archive per retention policy, or raise an approved quota-extension change.
3. Restore: locate the snapshot/backup nearest the last-seen time, restore to a side location, have the user verify integrity before overwriting anything.
4. Performance: check volume utilisation and backup windows overlapping business hours; reschedule jobs if so.

## Escalation
Escalate to Storage/Backup L2 for failed restores, replication issues, or capacity expansion.
