package com.vision.app.core.ml.liveness

import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * Head pose from MediaPipe-style 6 landmark indices, with solvePnP-style
 * closed-form approximation (without OpenCV).
 */
class HeadPoseEstimator {
    private val model3D = arrayOf(
        floatArrayOf(0f, 0f, 0f),
        floatArrayOf(0f, -63.6f, -12.5f),
        floatArrayOf(0f, 63.6f, -12.5f),
        floatArrayOf(-43.3f, 32.7f, -26f),
        floatArrayOf(43.3f, 32.7f, -26f),
        floatArrayOf(-21f, 0f, -30f),
        floatArrayOf(21f, 0f, -30f),
    )
    private val indices = intArrayOf(1, 152, 10, 454, 234, 133, 362)

    private val yawBuf = ArrayDeque<Float>()
    private val pitchBuf = ArrayDeque<Float>()
    private val rollBuf = ArrayDeque<Float>()
    private val samples = ArrayDeque<Triple<Float, Float, Float>>()
    private val smooth = 3

    fun update(points: Array<FloatArray>, w: Int, h: Int): Triple<Float, Float, Float> {
        val fx = w.toFloat()
        val pts2d = Array(7) { i ->
            val p = points[indices[i]]
            floatArrayOf(p[0], p[1])
        }
        // Approximate yaw/pitch from face geometry (no full PnP).
        val leftEye = points[indices[5]]
        val rightEye = points[indices[6]]
        val nose = points[indices[0]]
        val chin = points[indices[1]]
        val dx = (rightEye[0] - leftEye[0])
        val yaw = ((nose[0] - (leftEye[0] + rightEye[0]) / 2f) / (dx + 1e-3f)) * 90f
        val pitch = ((nose[1] - chin[1]) / (h + 1e-3f)) * -90f
        val roll = kotlin.math.atan2(rightEye[1] - leftEye[1], dx) * 57.2958f
        if (yawBuf.size == smooth) yawBuf.removeFirst()
        if (pitchBuf.size == smooth) pitchBuf.removeFirst()
        if (rollBuf.size == smooth) rollBuf.removeFirst()
        yawBuf.addLast(yaw); pitchBuf.addLast(pitch); rollBuf.addLast(roll)
        if (samples.size > 200) samples.removeFirst()
        samples.addLast(Triple(yaw, pitch, roll))
        val ya = yawBuf.average().toFloat()
        val pa = pitchBuf.average().toFloat()
        val ra = rollBuf.average().toFloat()
        return Triple(ya, pa, ra)
    }

    val score: Float
        get() {
            if (samples.size < 2) return 0f
            val yaws = samples.map { it.first }
            val pitches = samples.map { it.second }
            val rolls = samples.map { it.third }
            val ys = yaws.max() - yaws.min()
            val ps = pitches.max() - pitches.min()
            val rs = rolls.max() - rolls.min()
            return (0.5f * (ys / 8f).coerceAtMost(1f)
                + 0.3f * (ps / 5f).coerceAtMost(1f)
                + 0.2f * (rs / 3f).coerceAtMost(1f))
        }

    fun reset() {
        yawBuf.clear(); pitchBuf.clear(); rollBuf.clear(); samples.clear()
    }
}
