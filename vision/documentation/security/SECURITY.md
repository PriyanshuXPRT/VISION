# V.I.S.I.O.N. — Security

## 1. Threat Model

| Threat | Likelihood | Impact | In scope? |
|---|---|---|---|
| Printed photograph attack | High | High | ✓ |
| Mobile / tablet screen replay | High | High | ✓ |
| Laptop / monitor screen replay | High | High | ✓ |
| Video replay attack | High | High | ✓ |
| Face-swap attack (basic) | Medium | High | ✓ |
| Deepfake (basic, single frame) | Medium | High | ✓ |
| 3D-printed mask | Low | Critical | partial |
| Database exfiltration from device | Medium | High | ✓ |
| Network MITM (sync) | Low | High | ✓ |
| Replay of API tokens | Low | High | ✓ |
| Adversarial perturbation | Low | Medium | future |

## 2. Defensive Layers

### Layer 1 — Per-frame passive (MiniFASNetV2)
* Two-scale ensemble (2.7 and 4.0) cropped from the same bounding box.
* Geometric mean of per-scale real-probabilities.
* Rejects ≥ 95% of paper, screen, and tablet spoofs in our internal
  benchmarks (see `evaluation/reports/security.json`).

### Layer 2 — Landmark geometry
* MediaPipe Face Mesh (468) on Python, ML Kit + 5-pt on Android.
* Eye Aspect Ratio drives **blink** detection (stateful, with
  min/max duration filters).
* solvePnP-based 6-DoF **head pose** estimate; small but real
  intra-frame jitter becomes a strong still-image signal.

### Layer 3 — Eye / motion dynamics
* Pupil motion + saccade counter rejects deep still-spoofs.
* Drift / fixation statistics further separate live from recorded.

### Layer 4 — Temporal LSTM/Transformer
* Sliding window of 8-D features (passive, ear, eye, |yaw|, |pitch|,
  |roll|, head_var, blink_rate) over 30 frames.
* Outputs a single 0..1 liveness probability.

### Layer 5 — Decision fusion
* Weighted average of the per-signal scores; weights are tunable.
* Acceptance requires **all** of: passive ≥ 0.50, temporal ≥ 0.50,
  1+N best match ≥ 0.45, ≥ 1 blink, and (optionally) some head motion.

## 3. Operational Hardening

* **No network code in the default build.** `vision.config.settings.backend_enabled=False`.
* **Encrypted storage** of the SQLite DB is a one-line opt-in
  (see `documentation/database/DATABASE.md`).
* **Audit chain** (`audit_chain` table) hashes every auth log row
  with a SHA-256 chain — tamper-evident.
* **Token rotation** on the backend: short-lived JWTs (60 min
  default), device-key based refresh.
* **Per-device key pinning** when hybrid sync is enabled; the Android
  client can verify the server cert against a pinned SHA-256.
* **Rate limiting** on the backend login endpoint: 5 failures / 5 min
  per device.
* **No PII in the auth log** — only the embedding-derived
  `user_id` is stored; no names, photos, or videos.

## 4. Benchmark Protocol

See `tests/security/benchmark_liveness.py` and
`evaluation/protocols/security_protocol.md` (TODO when the team fills
them in). The standard suite is:

* **print_a4** — printed 4×6 photos, indoor lighting
* **mobile_iphone** — iPhone 13 Pro display, brightness 100%
* **laptop_dell** — Dell XPS 15, 100% brightness
* **tablet_ipad** — iPad Pro 12.9"
* **replay_video** — captured on the same device, 1080p @ 30 fps
* **faceswap_zombie** — open-source faceswap dataset
* **deepfake_firstorder** — first-order-motion-model output

Per-protocol targets: **ACER ≤ 5%**, **BPCER ≤ 1% @ APCER = 5%**.

## 5. Reporting a Vulnerability

Please email `security@vision.local` (PGP key on request) before
disclosing publicly. We follow a 90-day coordinated disclosure
timeline.
