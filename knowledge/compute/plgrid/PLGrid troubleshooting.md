---
title: PLGrid troubleshooting
type: handbook
domain: plgrid
tags: [handbook, plgrid, troubleshooting]
created: 2026-10-11
updated: 2026-10-11
parent: "[[PLGrid compute]]"
---
# PLGrid troubleshooting

## Errors and fixes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Cannot log in although a grant exists | The access service for the cluster is not active | Activate it in the Services tab ([[PLGrid access and grants]]) |
| `invalid login` or `invalid password` | Wrong login (it needs the `plg` prefix) or a forgotten password | Reset through the Portal's "Forgot password" option |
| No home directory on a new account | The account has not propagated yet | Wait about an hour |
| `sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified` | Grant resources exhausted, or a wrong account suffix or partition | Check utilisation (`hpc-grants` on Ares and Athena), renegotiate or apply for a new grant ([[PLGrid Slurm]]) |
| `FATAL: permission denied` running a script | Missing execute bit | `chmod u+x <script>` |
| Jobs wait very long | Queue priority and the requested resources | `sbatch --test-only`, shorter time limit, smaller request |
| Grant blocked | 100 % of the allocation used, or use against the stated purpose | Use another active grant or apply early for a new one |
| Affiliation inactive | It was not extended in time | Extend it in the Portal or ask the Helpdesk |

## Reporting to the Helpdesk

Use <https://helpdesk.plgrid.pl/> with a Portal account, otherwise email the operator address printed on that page; most reports are answered within 24 hours. Include:

- how the problem shows itself and the exact error text,
- the grant ID, the cluster and the application,
- the job IDs of failing jobs and the command and launch script used,
- input and output data or screenshots when relevant.

Name and surname changes and software installation requests also go through the Helpdesk.
