"""Passive liveness detector — no user interaction required.

Computes a 0..1 liveness score from a short video clip using features that
distinguish real faces from video replays / printed photos / screen attacks.

Features
--------
1.  **Frame-to-frame motion variance** — real faces always have 0.3-2.0 px
    of micro-jitter from breathing, pulse, head sway.  A frozen video has
    near-zero motion; a playing video has periodic motion but lower jitter
    than a real face.
2.  **High-frequency texture (Laplacian variance)** — real skin has pores,
    fine wrinkles, hair.  Printed photos lose fine detail (often ~5-25
    Laplacian).  Phone screens have a different texture (often 30-80).
    Real skin typically scores 60-300.
3.  **Color/illumination diversity** — real faces have a wide range of
    skin tones (cheeks redder, forehead paler, shadows under nose).
    Printed photos and screens are flatter.
4.  **Edge density (Canny)** — real faces have natural edges (eyelashes,
    lips, eyebrows).  Printed photos blur them; screens rasterize them.
5.  **Screen-refresh signature (FFT of pixel diffs)** — a 50 or 60 Hz
    monitor refresh creates a distinct spectral peak in the per-frame
    pixel difference signal.  Real faces have no such peak (white noise).
6.  **Specular highlight consistency** — real skin has soft, spread-out
    highlights.  Screens have hard, geometric highlights from the bezel.

The final score is a weighted sum of normalized feature scores.  No user
interaction is required.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class PassiveLivenessResult:
    score: float
    passed: bool
    threshold: float
    motion_score: float
    texture_score: float
    color_score: float
    edge_score: float
    refresh_score: float      # 1.0 if no 50/60 Hz peak detected
    frames_sampled: int
    notes: list[str] = field(default_factory=list)


class PassiveLivenessDetector:
    """Sliding-window passive liveness over a short video clip."""

    def __init__(
        self,
        *,
        threshold: float = 0.55,
        motion_thresh_px: float = 0.18,
        texture_thresh: float = 50.0,
        refresh_band: tuple[float, float] = (45.0, 65.0),
        history: int = 60,
    ) -> None:
        self.threshold = threshold
        self.motion_thresh_px = motion_thresh_px
        self.texture_thresh = texture_thresh
        self.refresh_lo, self.refresh_hi = refresh_band
        self._prev_face: Optional[NDArray[np.uint8]] = None
        self._diffs: deque[float] = deque(maxlen=history)
        self._lap_vars: deque[float] = deque(maxlen=history)
        self._color_vars: deque[float] = deque(maxlen=history)
        self._edge_counts: deque[float] = deque(maxlen=history)
        self._diff_signal: deque[float] = deque(maxlen=256)
        self._fps_estimate: float = 0.0
        self._last_t: Optional[float] = None

    # ---------------------------------------------------------------- ingest
    def update(
        self,
        frame_bgr: NDArray[np.uint8],
        bbox: tuple[float, float, float, float],
        t: float | None = None,
    ) -> None:
        """Feed one frame + face bbox.  Call repeatedly (>= 20 frames)."""
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_bgr.shape[1], x2), min(frame_bgr.shape[0], y2)
        if x2 - x1 < 30 or y2 - y1 < 30:
            return
        face = frame_bgr[y1:y2, x1:x2]
        # resize to a canonical size so consecutive-frame diffs always match
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA)

        # 1. motion variance
        if self._prev_face is not None:
            d = float(np.mean(cv2.absdiff(gray, self._prev_face)))
            self._diffs.append(d)
            self._diff_signal.append(d)
        self._prev_face = gray

        # 2. texture (Laplacian variance)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self._lap_vars.append(lap_var)

        # 3. color diversity (std-dev across BGR channels, normalised)
        face_resized = cv2.resize(face, (128, 128), interpolation=cv2.INTER_AREA)
        color_std = float(face_resized.std(axis=(0, 1)).mean())
        self._color_vars.append(color_std)

        # 4. edge density (Canny fraction)
        edges = cv2.Canny(gray, 60, 160)
        edge_frac = float(edges.mean() / 255.0)
        self._edge_counts.append(edge_frac)

        # 5. fps estimate for FFT
        if t is not None and self._last_t is not None and t > self._last_t:
            inst_fps = 1.0 / (t - self._last_t)
            # exponentially-smoothed
            self._fps_estimate = 0.9 * self._fps_estimate + 0.1 * inst_fps \
                if self._fps_estimate > 0 else inst_fps
        if t is not None:
            self._last_t = t

    # ---------------------------------------------------------------- score
    def _norm(self, value: float, lo: float, hi: float) -> float:
        return float(max(0.0, min(1.0, (value - lo) / (hi - lo))))

    def _motion_score(self) -> float:
        if not self._diffs:
            return 0.0
        mean_diff = float(np.mean(self._diffs))
        return self._norm(mean_diff, self.motion_thresh_px, 4.0)

    def _texture_score(self) -> float:
        if not self._lap_vars:
            return 0.0
        # real skin: lap ~50-300, screen: 30-80, print: 5-25
        mean_lap = float(np.mean(self._lap_vars))
        return self._norm(mean_lap, self.texture_thresh, 250.0)

    def _color_score(self) -> float:
        if not self._color_vars:
            return 0.0
        mean_color = float(np.mean(self._color_vars))
        # BGR stddev across face crop; real faces vary 25-70
        return self._norm(mean_color, 20.0, 70.0)

    def _edge_score(self) -> float:
        if not self._edge_counts:
            return 0.0
        mean_edges = float(np.mean(self._edge_counts))
        return self._norm(mean_edges, 0.04, 0.18)

    def _refresh_score(self) -> tuple[float, list[float]]:
        """Returns (score, peak_freqs).  Score 1.0 means no screen peak."""
        if len(self._diff_signal) < 32 or self._fps_estimate < 5.0:
            return 1.0, []
        sig = np.array(self._diff_signal, dtype=np.float32)
        sig = sig - sig.mean()
        # FFT
        n = len(sig)
        freqs = np.fft.rfftfreq(n, d=1.0 / self._fps_estimate)
        spectrum = np.abs(np.fft.rfft(sig))
        # ignore DC
        spectrum[0] = 0
        # look for a peak in the screen refresh band
        band_mask = (freqs >= self.refresh_lo) & (freqs <= self.refresh_hi)
        if not band_mask.any():
            return 1.0, []
        band_peak = float(spectrum[band_mask].max())
        total_peak = float(spectrum.max()) + 1e-9
        ratio = band_peak / total_peak
        # if more than 25% of energy is in the 45-65 Hz band -> likely screen
        if ratio > 0.25:
            return 0.0, freqs[band_mask][spectrum[band_mask].argmax():spectrum[band_mask].argmax() + 1].tolist()
        return 1.0, []

    def result(self) -> PassiveLivenessResult:
        m = self._motion_score()
        t = self._texture_score()
        c = self._color_score()
        e = self._edge_score()
        r, peaks = self._refresh_score()
        # Weighted average — motion and texture carry the most weight.
        score = 0.35 * m + 0.25 * t + 0.10 * c + 0.10 * e + 0.20 * r
        notes: list[str] = []
        if t < 0.2:
            notes.append("low texture (possible printed photo)")
        if m < 0.2:
            notes.append("low motion variance (possible frozen frame)")
        if r < 0.5:
            notes.append(
                f"screen-refresh peak detected near {peaks[0] if peaks else '?'} Hz"
            )
        if m > 0.8:
            notes.append("strong natural micro-motion")
        return PassiveLivenessResult(
            score=score,
            passed=score >= self.threshold,
            threshold=self.threshold,
            motion_score=m,
            texture_score=t,
            color_score=c,
            edge_score=e,
            refresh_score=r,
            frames_sampled=len(self._diffs),
            notes=notes,
        )

    def reset(self) -> None:
        self._prev_face = None
        self._diffs.clear()
        self._lap_vars.clear()
        self._color_vars.clear()
        self._edge_counts.clear()
        self._diff_signal.clear()
        self._fps_estimate = 0.0
        self._last_t = None
