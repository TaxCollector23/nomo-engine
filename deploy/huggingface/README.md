---
title: Nomo Backend
emoji: 🧠
colorFrom: gray
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Nomo tri-domain HW-NAS search + live telemetry API
---

# Nomo backend

FastAPI + WebSocket telemetry server for the Nomo dashboard.

- Health: `/healthz`
- Catalog: `/catalog`
- Live stream: `wss://<this-space>.hf.space/ws/runs/<run_id>`

Point the dashboard at it with `NEXT_PUBLIC_NOMO_API=https://<user>-<space>.hf.space`.
