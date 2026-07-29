# scripts/generate_tickets.py
"""Generate the OpsMedic synthetic historical-ticket knowledge base.

Written from scratch for this project. Produces data/tickets.csv with a
deterministic (seeded) mix of realistic L1/L2 IT incidents across six
categories. These tickets are the RAG corpus that sub-task 1 retrieves
from; they are NOT the fine-tuning dataset (that is a cited public
dataset, see finetune/data.py).

Usage:
    python scripts/generate_tickets.py            # 200 rows (default)
    python scripts/generate_tickets.py --rows 500 --out data/tickets.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from datetime import date, timedelta

# --------------------------------------------------------------- templates
# Each category: list of (title, description, resolution_steps) templates.
# {x} placeholders are filled from the pools below.
TEMPLATES: dict[str, list[tuple[str, str, str]]] = {
    "Network": [
        (
            "VPN connection drops for {dept} user",
            "User in {dept} reports VPN disconnects every few minutes on {os}. "
            "Error shown: gateway timeout. Home ISP is {isp}.",
            "1. Confirm VPN client version and update if below baseline.\n"
            "2. Switch VPN protocol from UDP to TCP in client settings.\n"
            "3. Whitelist VPN executable in local firewall/antivirus.\n"
            "4. If drops persist, capture client logs and escalate to Network L2.",
        ),
        (
            "Cannot reach internal portal {app}",
            "Multiple users report {app} unreachable from office LAN, "
            "browser shows DNS_PROBE_FINISHED_NXDOMAIN. Wi-Fi segment {vlan}.",
            "1. Verify DNS resolution with nslookup against corporate DNS.\n"
            "2. Flush client DNS cache (ipconfig /flushdns).\n"
            "3. Check VLAN {vlan} DNS forwarder configuration.\n"
            "4. Raise change request to correct forwarder entry if wrong.",
        ),
        (
            "Slow network in {site} office",
            "Users at {site} site report very slow file transfers and video "
            "calls dropping since morning. Approx {num} users affected.",
            "1. Check WAN link utilisation on the {site} edge router.\n"
            "2. Identify top talkers; look for backup jobs running in office hours.\n"
            "3. Reschedule backup job to off-peak window.\n"
            "4. Monitor link for 2 hours and confirm with site contact.",
        ),
    ],
    "Access": [
        (
            "Account locked after failed logins",
            "User {user} locked out of AD account after multiple failed "
            "password attempts on {os}. Needs urgent access for {dept} work.",
            "1. Verify user identity per security policy (manager + employee ID).\n"
            "2. Unlock account in Active Directory Users and Computers.\n"
            "3. Reset password and enforce change at next logon.\n"
            "4. Advise on password manager to prevent recurrence.",
        ),
        (
            "Access request for {app} denied",
            "User {user} from {dept} cannot open {app}; gets 403 Forbidden "
            "although manager approved access last week.",
            "1. Check user's AD group membership against {app} access matrix.\n"
            "2. Confirm the approval ticket completed provisioning workflow.\n"
            "3. Add user to the correct entitlement group.\n"
            "4. Ask user to re-login after 15 minutes; confirm access.",
        ),
        (
            "MFA token not received",
            "User {user} does not receive MFA push/SMS on registered device "
            "after switching to a new phone. Cannot log in to any SSO app.",
            "1. Verify identity per policy before any MFA change.\n"
            "2. Remove old device registration from the MFA portal.\n"
            "3. Guide user through re-enrolling the new device.\n"
            "4. Test SSO login to one app and confirm closure.",
        ),
    ],
    "Hardware": [
        (
            "Laptop not powering on",
            "{dept} user's {make} laptop shows no power light; was working "
            "yesterday. Charger LED is {led}.",
            "1. Try a known-good charger and power outlet.\n"
            "2. Perform hard reset: hold power 30 seconds without battery/AC.\n"
            "3. If still dead, book replacement device from stock.\n"
            "4. Raise RMA with vendor for the faulty unit.",
        ),
        (
            "Printer {asset} not printing",
            "Shared printer {asset} on floor {num} shows jobs queued but "
            "nothing prints; panel displays no error.",
            "1. Restart the print spooler service on the print server.\n"
            "2. Clear stuck jobs from the queue.\n"
            "3. Power-cycle the printer and reseat network cable.\n"
            "4. Print test page; if failing, swap with spare unit and log RMA.",
        ),
        (
            "Docking station display issues",
            "User's external monitors flicker and disconnect when laptop is "
            "docked. Direct HDMI to laptop works fine. Dock model {make}.",
            "1. Update dock firmware and laptop USB-C/Thunderbolt drivers.\n"
            "2. Replace the dock's upstream cable with a certified cable.\n"
            "3. Test with a known-good dock to isolate the unit.\n"
            "4. Replace dock from stock if fault follows the unit.",
        ),
    ],
    "Software": [
        (
            "{app} crashes on startup",
            "After the latest update, {app} crashes immediately on launch "
            "for user in {dept} on {os}. Event viewer shows faulting module.",
            "1. Clear the application cache/user profile settings folder.\n"
            "2. Repair-install the application from the software portal.\n"
            "3. If crash persists, roll back to previous version.\n"
            "4. Log a problem record referencing the faulting module for vendor.",
        ),
        (
            "License activation failure for {app}",
            "User receives 'license server unreachable' when starting {app}. "
            "Approx {num} users in {dept} affected simultaneously.",
            "1. Check license server service status and restart if stopped.\n"
            "2. Verify license count not exhausted; reclaim idle seats.\n"
            "3. Confirm client can reach license server port (telnet test).\n"
            "4. Notify affected users and monitor for one business day.",
        ),
        (
            "Email stuck in Outbox",
            "User on {os} reports emails with attachments stay in Outbox; "
            "small emails send fine. Mailbox size near {num} GB.",
            "1. Check attachment size against the send limit.\n"
            "2. Archive old mail to bring mailbox under quota.\n"
            "3. Recreate the Outlook profile / clear OST cache.\n"
            "4. Send test email with attachment and confirm delivery.",
        ),
    ],
    "Database": [
        (
            "Nightly job failed on {db} database",
            "The ETL batch job on {db} failed at step LOAD with deadlock "
            "error. Downstream {app} reports are stale this morning.",
            "1. Check job logs for the deadlocked session and victim query.\n"
            "2. Re-run the failed step after confirming no blocking sessions.\n"
            "3. Add retry-on-deadlock logic to the job (change request).\n"
            "4. Validate downstream {app} report freshness with data team.",
        ),
        (
            "Slow queries on {db} since morning",
            "Application team reports {app} timeouts; DB {db} CPU at high "
            "utilisation. A deployment happened last night.",
            "1. Identify top queries by CPU/reads since the deployment.\n"
            "2. Compare execution plans against pre-deployment baseline.\n"
            "3. Update statistics / rebuild the affected index.\n"
            "4. If regression confirmed, roll back the offending query change.",
        ),
        (
            "Database {db} storage almost full",
            "Monitoring alert: data volume for {db} at 92% capacity and "
            "growing ~{num} GB per day.",
            "1. Purge/archive per data-retention policy with app-team sign-off.\n"
            "2. Shrink transaction log after log backup if applicable.\n"
            "3. Extend the volume via storage change request.\n"
            "4. Set early-warning alert at 80% to avoid recurrence.",
        ),
    ],
    "Security": [
        (
            "Phishing email reported by {dept}",
            "User {user} reports suspicious email asking to update payroll "
            "details via external link. Several colleagues received it too.",
            "1. Quarantine the message org-wide via mail security console.\n"
            "2. Block sender domain and the embedded URL at the gateway.\n"
            "3. Check click-through logs; force password reset for clickers.\n"
            "4. Send user advisory and log the incident with SecOps.",
        ),
        (
            "Endpoint flagged with malware on {os}",
            "AV console flags trojan on {dept} user's machine {asset}; "
            "device auto-isolated from network.",
            "1. Keep device isolated; collect AV detection details.\n"
            "2. Run full offline scan and remove the detected threat.\n"
            "3. Verify no lateral movement from the endpoint in EDR logs.\n"
            "4. Re-image if persistence is suspected; restore user data from backup.",
        ),
        (
            "Suspicious login alert for {user}",
            "SIEM alert: impossible-travel login for {user} — office login "
            "followed by foreign-country login within minutes.",
            "1. Disable the account immediately pending verification.\n"
            "2. Contact user via phone to verify recent activity.\n"
            "3. Reset credentials and revoke active sessions/tokens.\n"
            "4. Review mailbox rules and audit logs for tampering.",
        ),
    ],
}

POOLS: dict[str, list[str]] = {
    "dept": ["Finance", "HR", "Sales", "Engineering", "Procurement", "Legal"],
    "os": ["Windows 11", "Windows 10", "macOS 14", "Ubuntu 22.04"],
    "isp": ["Airtel", "Jio", "ACT", "BSNL"],
    "app": ["Timesheet", "Expensify", "SAP GUI", "Confluence", "Tableau", "CRM"],
    "vlan": ["VLAN-20", "VLAN-31", "VLAN-42"],
    "site": ["Delhi", "Bengaluru", "Hyderabad", "Pune", "Noida"],
    "user": ["a.sharma", "p.iyer", "r.gupta", "s.khan", "v.rao", "m.das"],
    "make": ["Dell Latitude", "Lenovo ThinkPad", "HP EliteBook", "MacBook Pro"],
    "led": ["on", "off", "blinking"],
    "asset": ["PRN-104", "PRN-207", "AST-5521", "AST-7310"],
    "db": ["ORDERSDB", "HRMSDB", "FINDB", "CRMDB"],
    "num": ["6", "12", "25", "40"],
}


def _fill_templates(
    templates: tuple[str, str, str], rng: random.Random
) -> tuple[str, str, str]:
    """Fill placeholders consistently across title/description/resolution.

    One value per placeholder key is drawn per ticket, so e.g. {site} is
    the same city in the title, the description, and the fix steps.
    """
    mapping = {key: rng.choice(values) for key, values in POOLS.items()}

    def _sub(text: str) -> str:
        for key, value in mapping.items():
            text = text.replace("{" + key + "}", value)
        return text

    title, desc, res = templates
    return _sub(title), _sub(desc), _sub(res)


def generate(rows: int, seed: int = 42) -> list[dict[str, str]]:
    rng = random.Random(seed)
    categories = list(TEMPLATES.keys())
    start = date(2024, 1, 1)
    records: list[dict[str, str]] = []
    for i in range(1, rows + 1):
        cat = categories[(i - 1) % len(categories)]  # balanced classes
        title, desc, res = _fill_templates(rng.choice(TEMPLATES[cat]), rng)
        resolved = start + timedelta(days=rng.randint(0, 540))
        records.append(
            {
                "ticket_id": f"OPM-{i:04d}",
                "title": title,
                "description": desc,
                "category": cat,
                "priority": rng.choice(["P1", "P2", "P3", "P3", "P4"]),
                "resolution_steps": res,
                "resolved_date": resolved.isoformat(),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--out", default="data/tickets.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = generate(args.rows, args.seed)
    fieldnames = list(records[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} tickets -> {args.out}")
    by_cat: dict[str, int] = {}
    for r in records:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat:<10} {count}")


if __name__ == "__main__":
    main()
