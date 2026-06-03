package com.vision.app.core.ml.liveness

import kotlin.math.hypot

/**
 * Eye Aspect Ratio + blink counter. Mirrors the Python `BlinkDetector`.
 */
class BlinkDetector(
    private val earThresh: Float = 0.21f,
    private val consecFrames: Int = 2,
    private val minBlinkMs: Long = 80,
    private val maxBlinkMs: Long = 600,
) {
    private val leftEye = intArrayOf(33, 160, 158, 133, 153, 144)
    private val rightEye = intArrayOf(362, 385, 387, 263, 373, 380)

    private var below = 0
    private var blinkStart: Long? = null
    private var blinkMin = 1f
    var blinkCount: Int = 0
        private set

    fun update(points: Array<FloatArray>, tsMs: Long): Float {
        val l = ear(leftEye, points)
        val r = ear(rightEye, points)
        val ear = 0.5f * (l + r)
        if (ear < earThresh) {
            if (blinkStart == null) { blinkStart = tsMs; blinkMin = ear }
            blinkMin = minOf(blinkMin, ear)
            below++
        } else {
            val s = blinkStart
            if (s != null && below >= consecFrames) {
                val dur = tsMs - s
                if (dur in minBlinkMs..maxBlinkMs) blinkCount++
            }
            blinkStart = null
            below = 0
            blinkMin = 1f
        }
        return ear
    }

    val score: Float
        get() = (minOf(blinkCount, 3) / 3f).coerceIn(0f, 1f)

    fun reset() {
        below = 0
        blinkStart = null
        blinkMin = 1f
        blinkCount = 0
    }

    private fun ear(idxs: IntArray, pts: Array<FloatArray>): Float {
        val p1 = pts[idxs[0]]; val p2 = pts[idxs[1]]; val p3 = pts[idxs[2]]
        val p4 = pts[idxs[3]]; val p5 = pts[idxs[4]]; val p6 = pts[idxs[5]]
        val a = hypot(p2[0]-p6[0], p2[1]-p6[1])
        val b = hypot(p3[0]-p5[0], p3[1]-p5[1])
        val c = 2f * hypot(p1[0]-p4[0], p1[1]-p4[1]) + 1e-6f
        return (a + b) / c
    }
}
