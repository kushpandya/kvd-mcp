---
title: "Docker Networking Guide"
tags: ["docker", "devops", "networking"]
created: 2026-07-30
---

# Docker Networking Guide

## Bridge Networks
By default, Docker creates a bridge network for containers. This allows them to communicate with each other on the same host.

## Host Networking
When you use `--network host`, the container shares the host's networking namespace. This is useful for performance but reduces isolation.

## Troubleshooting
If containers cannot resolve DNS, check the `/etc/resolv.conf` file inside the container.
