# Security policy

## Private vulnerability reports

Use GitHub's private vulnerability reporting form:
[Report a vulnerability](https://github.com/XinchengSun/OpenVA-Dialogue/security/advisories/new).
Do not put credentials, exploit details against a live service, or private user
data in public issues or pull requests. If the form is unavailable, request a
private reporting channel from the maintainer without posting sensitive details.

Include the affected commit/profile, impact, minimal reproduction on a system
you control, and sanitized evidence. Avoid scanning, exploiting, or interrupting
other people's deployments. If a token has already leaked, revoke/rotate it
at its provider; deleting a GitHub message does not invalidate the credential.

## Supported scope

Security fixes are considered for the current `main` branch. Older snapshots
have no separate support commitment. This is a research system, not a hardened
multi-tenant service, and no response-time SLA is promised. Provider services,
model runtimes, browsers, and dependencies also require their own updates.

## Deployment precautions

- Prefer a loopback bind plus a local SSH tunnel for private development.
- Public deployments need HTTPS and the configured access controls; do not
  disable public/admin authentication to fix a login error.
- Avatar customization replaces shared runtime state. Restrict administrator
  access to trusted people and protect concurrency/resource admission controls.
- Keep API keys, tokens, reference portraits/audio, and transcripts outside Git.
  Redact cookies, authorization headers, query tokens, and private paths in logs.
- Microphone audio/text can reach the configured ASR/LLM/TTS providers. Review
  that data flow with users; the single-GPU profile still uses an LLM API.
- Obtain permission for uploaded portraits and cloned voices. Do not publish
  identity-bearing demo media without the relevant rights and consent.
