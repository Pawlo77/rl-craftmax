---
title: PLGrid access and grants
type: handbook
domain: plgrid
tags: [handbook, plgrid, grants]
created: 2026-10-11
updated: 2026-10-11
parent: "[[PLGrid compute]]"
---
# PLGrid access and grants

## Path to a running job

1. **Account and affiliation.** Sign up in the PLGrid Portal and declare an affiliation with a Polish research, administration or industry unit; wait for the affiliation to be approved. Only an active affiliation can use a grant.
2. **Team.** Every grant needs a team, which may have one member. Only users with an academic, scientific-institution, PhD-student or HPC-centre-employee affiliation can create one. Subordinates and external collaborators join a team their supervisor created.
3. **Grant.** The team's main manager applies for it in the Grants tab.
4. **Service.** Activate the access service for the target cluster in the Services tab; it may need manager confirmation and may require a grant with resources on that cluster. A new account needs about an hour before its home directory appears on the cluster.
5. **Login and compute.** SSH to the access node ([[PLGrid clusters]]) and submit through [[PLGrid Slurm]].
6. **Settle.** Report publications after the grant ends.

## Roles

| Role | Can |
|------|-----|
| Main manager | Invite members, apply for and renegotiate grants, owns settlement; cannot leave the team before handing the role over |
| Manager | Invite members, edit applications, renegotiate; shares responsibility for settlement |
| Member | Use the allocated resources; no team or grant management |

## Grant types

| Type | Facts |
|------|-------|
| Pilot | One per 12 months, one active at a time; 1000 CPU hours and 10 GB storage by default; max job runtime 72 h; active within an hour; resources must be used within a year. Needs a real research topic, not infrastructure testing. When exhausted, renegotiate instead of applying again |
| Proper | Negotiable resource configuration; needs research purpose, computational readiness, related projects and declared publications; takes up to 7 working days |
| Academic | For teaching and student classes, only for academic-unit employees; the call opens twice a year for one semester |

A pilot grant defaults to CPU time and storage; other resource types, such as GPU, are added with the plus buttons or through renegotiation.

## Renegotiation, blocking, settlement

- **Renegotiation** adds resources or a new resource type to an active grant, takes up to 5 working days, and cannot extend the duration or end the grant early. Only the applicant, as team manager, can do it. Separately, resource changes proposed to a user must be accepted within 30 days or the application may be deleted.
- **Blocking** happens at 100 % use of the allocation or use against the stated purpose. A block can apply to one centre or all of them. Continue with another active grant or a new one applied for well in advance. On Ares and Athena the `hpc-grants` command shows utilisation.
- **Settlement** is due within a month of the grant's end, in the Settlements tab: add the resulting publications or declare them for later, add research-data repositories, and fill the satisfaction survey. Drafts, submitted-only papers and arXiv preprints do not count; a paper is added once it is published with a DOI. One publication belongs to one grant only; a series of continued grants can be settled with the publications on the last one.
- **Follow-up grant:** mark it as a continuation of the previous one.
- **Acknowledgement** of PLGrid and the computing centre, with the grant number, is required in every publication, thesis and conference paper. The exact wording and the logo are in the grant's Settlement tab and on the guide's Acknowledgments page.

## Cost

PLGrid is free for academic users; the allocation is decided by expert review of the application. Industry users are the exception.
