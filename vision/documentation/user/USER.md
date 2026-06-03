# V.I.S.I.O.N. — User Manual

## Getting started

1. Install **V.I.S.I.O.N.** from your enterprise MDM or the
   [release page](../../releases).
2. Open the app. The first time, you will see the splash screen and
   then the **Login** screen.
3. Choose **Authenticate (Face)** to identify yourself, or
   **Admin Login** to access the management console.

## For end users (authentication)

1. Tap **Authenticate (Face)**.
2. Hold the phone 30–60 cm from your face, in a well-lit area.
3. The on-screen arc shows whether a face is detected.
4. Look at the camera and blink naturally. A real person will blink
   once or twice during the 3-second window.
5. Slowly turn your head a few degrees left and right.
6. After about 2–3 seconds the result appears at the bottom of the
   screen. **Green** = accepted, **Red** = rejected.

If authentication is rejected, common reasons are:

* **Lighting too low or too high** — move to a different spot.
* **Face too small** — bring the phone closer.
* **Looking away** — keep eyes towards the camera.
* **Wearing a heavy mask or sunglasses** — remove them and try again.

## For administrators

### Register a new user

1. From the **Admin Dashboard**, tap **Register user**.
2. Fill in name (required), phone, email, notes.
3. Tap **Continue to face enrollment**.
4. The device will capture a 3-second video. Instruct the user to
   blink and slowly turn their head.
5. When enrollment finishes, the new user is added and a template is
   stored locally.

### Manage users

* **User management** lists all users with quick delete.
* **Auth logs** shows the last 100 attempts; tap a row for details
  (liveness, similarity, spoof classification).
* **Models** displays the loaded ONNX model versions and their
  SHA-256 hashes; the expected hashes are pinned in
  `models/registry/registry.json`.

### Settings

* **Hybrid sync** — when on, encrypted templates are pushed/pulled
  against the configured backend. Defaults to **off**.
* **Liveness threshold** — higher = stricter; default 0.85.
* **Similarity threshold** — higher = stricter; default 0.45.
* **Require blink / eye motion** — toggle active-liveness signals.

## Privacy

* All face templates and logs stay on the device by default.
* Hybrid sync is **opt-in**; when enabled, all traffic is over
  TLS 1.3 with certificate pinning.
* V.I.S.I.O.N. does not collect telemetry. No analytics, no
  crash reporting by default.
* To wipe all data, go to **Settings → Advanced → Wipe local data**
  (this also revokes the device key on the backend).
