---
title: PLGrid clusters
type: handbook
domain: plgrid
tags: [handbook, plgrid, clusters]
created: 2026-10-11
updated: 2026-10-11
parent: "[[PLGrid compute]]"
---
# PLGrid clusters

## Hardware

| Cluster | Centre | Node type | Cores per node | Memory per node | Accelerators per node | OS | Login node |
|---------|--------|-----------|----------------|-----------------|-----------------------|----|------------|
| Ares | ACC Cyfronet AGH | CPU | 48 | 192 or 384 GB | none | Rocky Linux 8 | `ares.cyfronet.pl` |
| Ares | ACC Cyfronet AGH | GPU | 32 | 384 GB | 8 × V100 | Rocky Linux 8 | `ares.cyfronet.pl` |
| Athena | ACC Cyfronet AGH | GPU | 128 | 1024 GB | 8 × A100 40 GB | Rocky Linux 9 | `athena.cyfronet.pl` |
| Helios | ACC Cyfronet AGH | CPU | 192 | 384 or 768 GB | none | Rocky Linux 9 | `helios.cyfronet.pl` |
| Helios | ACC Cyfronet AGH | GPU | 288 | 480 GB | 4 × GH200 96 GB | Rocky Linux 9 | `helios.cyfronet.pl` |
| Lem | WCSS Wrocław | CPU | 128 | 1536 GB | none | AlmaLinux | `ui.wcss.pl` |
| Lem | WCSS Wrocław | GPU | 64 | 1006 GB | 4 × H100 96 GB | AlmaLinux | `ui.wcss.pl` |
| LUMI | CSC Finland | GPU | 64 | 512 GB | 4 × MI250X | SLES 15 SP5 | `lumi.csc.fi` |
| Tryton Plus | CI TASK Gdańsk | CPU | 48 | 192 GB | none | Rocky Linux 8 | `plgrid.tryton.task.gda.pl` |

Peak power: Ares 4 PFlops, Athena 7.7, Helios about 36 to 37, Lem 21, LUMI 550 and Tryton Plus 3.8 (the last three only in part for PLGrid users).

## Helios

- 98 304 AMD Zen4 cores, 31 680 Grace cores, 440 GH200 96 GB accelerators, 24 H100 94 GB accelerators and 18 PB of disk on a 200 GB/s Slingshot network.
- The GH200 nodes are aarch64 (Grace CPU), so a virtual environment built for x86 cannot run there.
- Allocation follows the EuroHPC PL access regulations. A Helios grant must fill in **Related projects** (section 5, point 4b) and name the HPC-centre co-operation (point 4c); if none is named, it is added at verification.

## Lem

CPU partition: 188 nodes, 23 808 cores. GPU partition: 76 nodes with 304 H100 connected by NVLink. Storage: 2 PB NFS, 2 PB Lustre (permanent and temporary), 6 PB S3 object storage and 1 PB local NVMe. Open OnDemand (a web interface for running applications and browsing files) is available after activating the "Local access to the Lem clusters" service.

## Rules for access machines

- Never run computations on the login node; it is shared. Write batch scripts there and submit them.
- Access needs the cluster's access service to be active ([[PLGrid access and grants]]).
- Login names carry the `plg` prefix, and the password is the PLGrid Portal password; SSH keys are the recommended alternative ([[PLGrid storage and transfer]]).
