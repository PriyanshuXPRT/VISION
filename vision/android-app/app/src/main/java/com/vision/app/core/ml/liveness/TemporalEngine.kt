package com.vision.app.core.ml.liveness

import kotlin.math.exp

/**
 * Sliding-window heuristic temporal liveness (mirrors the Python
 * `HeuristicEngine`). Keeps the last N feature vectors and emits a final
 * score from them.
 */
class TemporalEngine(
    private val seqLen: Int = 30,
    private val stride: Int = 2,
) {
    private val buf = ArrayDeque<FloatArray>(128)
    private val threshold: Float = 0.5f

    val featureDim: Int = 8

    fun reset() = buf.clear()

    fun isReady(): Boolean = buf.size >= seqLen

    fun push(feature: FloatArray) {
        require(feature.size == featureDim) { "feature dim must be $featureDim" }
        if (buf.size == buf.capacity) buf.removeFirst()
        buf.addLast(feature.copyOf())
    }

    data class Result(val score: Float, val isLive: Boolean, val passive: Float, val blink: Float, val eye: Float, val head: Float)

    fun score(): Result {
        if (!isReady()) return Result(0f, false, 0f, 0f, 0f, 0f)
        val s = FloatArray(seqLen)
        val ear = FloatArray(seqLen)
        val eye = FloatArray(seqLen)
        val head = FloatArray(seqLen)
        val blink = FloatArray(seqLen)
        for (i in 0 until seqLen) {
            val f = buf.elementAt(buf.size - seqLen + i)
            s[i] = f[0]; ear[i] = f[1]; eye[i] = f[2]; head[i] = f[6]; blink[i] = f[7]
        }
        val passive = s.average().toFloat()
        val blinkR = blink.average().toFloat()
        val eyeR = eye.average().toFloat()
        val headR = head.average().toFloat()
        val sBlink = (1f - exp(-2.5f * blinkR)).toFloat()
        val sHead = (1f - exp(-3.0f * headR)).toFloat()
        val sMotion = 0.5f * (sHead + eyeR)
        val final = (0.45f * passive
            + 0.15f * sBlink
            + 0.15f * eyeR
            + 0.15f * sHead
            + 0.10f * sMotion).coerceIn(0f, 1f)
        return Result(final, final >= threshold, passive, sBlink, eyeR, sHead)
    }
}
