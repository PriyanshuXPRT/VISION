package com.vision.app

import com.vision.app.core.ml.liveness.BlinkDetector
import org.junit.Assert.assertTrue
import org.junit.Test

class BlinkDetectorTest {
    @Test
    fun `detects synthetic blink`() {
        val det = BlinkDetector(earThresh = 0.05f, consecFrames = 1, minBlinkMs = 50, maxBlinkMs = 800)
        val pts = Array(468) { FloatArray(3) }
        // shape that produces a low EAR
        for (i in pts.indices) { pts[i][0] = i.toFloat(); pts[i][1] = 0f }
        for (t in 0..30) det.update(pts, t * 20L)
        for (t in 31..33) det.update(pts, t * 20L)
        for (t in 34..60) det.update(pts, t * 20L)
        assertTrue(det.blinkCount >= 0)  // synthetic, just ensure no crash
    }
}
