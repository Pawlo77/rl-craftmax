---
title: PLGrid storage and transfer
type: handbook
domain: plgrid
tags: [handbook, plgrid, storage]
created: 2026-10-11
updated: 2026-10-11
parent: "[[PLGrid compute]]"
---
# PLGrid storage and transfer

## Storage classes

| Class | Properties |
|-------|------------|
| Home | Per user, mounted on every system the user can reach; for configuration, source code and small files |
| Project space | On all PLGrid systems, usually Lustre, mounted on all compute nodes; for active projects needing high throughput |
| Scratch | Cluster-wide or node-local depending on the system; temporary data that the system may delete after inactivity or by policy |
| Object storage | S3-compatible buckets in two independent locations, reachable from outside the cluster; for transferring data between centres, results, public sharing, backups and CI artifacts. Applied for through the Cyfronet Object Storage service |

Capacity, performance and policy differ per system; the page of each supercomputer is authoritative. Lem lists its own pools in [[PLGrid clusters]].

## SSH keys

- Generate a pair locally with `ssh-keygen -t ed25519 -a 100`, protect the private key with a passphrase, and never share it.
- Put the public key on the cluster with `ssh-copy-id <login>@<host>`, or append it to `~/.ssh/authorized_keys` and set that file to mode 600.
- Password login uses the PLGrid Portal password; keys are the recommended alternative because passwords are exposed to brute force.

## Transfer

- Use `scp` or `rsync` over SSH; `scp -C -r` copies a directory with compression. WinSCP is the graphical option on Windows.
- Object storage and `rclone` (available as a module) suit large moves and transfers between centres.
- Results that must outlive scratch go to the Hub for this project.
