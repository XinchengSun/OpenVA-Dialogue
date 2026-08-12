# Current Demo Runbook

The authoritative Chinese runbook, technical route, change log, acceptance
numbers, and troubleshooting guide are in the repository root `README.md`.

Server:

```bash
cd "${FLASHAV2AV_ROOT:-$HOME/FlashAV2AV}"
bash scripts/run_demo.sh
```

Windows client:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\connect_demo.ps1 `
  -SshHost <host> -SshPort <port>
```

Only `DEMO_READY` after the full health check means the demo is ready. Keep one
browser tab open. Do not use the legacy SeedDuplex or experimental WebRTC
launchers for the current demo.
