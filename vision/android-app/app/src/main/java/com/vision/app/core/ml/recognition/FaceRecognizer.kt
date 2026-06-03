package com.vision.app.core.ml.recognition

import android.content.Context
import com.vision.app.core.ml.OnnxEngine
import kotlin.math.sqrt

/**
 * ArcFace 512-D embedder (Android port).
 *
 * Input:  (1, 3, 112, 112) float32 RGB, normalised to [-1, 1].
 * Output: (1, 512) float32, L2-normalised.
 */
class FaceRecognizer(
    context: Context,
    modelAsset: String = "models/arcface_r100.onnx",
) {
    private val engine = OnnxEngine.fromAsset(context, modelAsset)

    fun generateEmbedding(alignedRgb112: ByteArray): FloatArray {
        val floats = FloatArray(3 * 112 * 112)
        for (i in alignedRgb112.indices) floats[i] = ((alignedRgb112[i].toInt() and 0xFF) - 127.5f) / 127.5f
        val outs = engine.runFloat(
            input = engine.inputNames[0] to floats,
            shape = longArrayOf(1, 3, 112, 112),
        )
        val raw = outs.values.first().first()
        return l2Normalize(raw)
    }

    private fun l2Normalize(v: FloatArray): FloatArray {
        var n = 0f
        for (x in v) n += x * x
        n = sqrt(n.toDouble()).toFloat().coerceAtLeast(1e-12f)
        val out = FloatArray(v.size)
        for (i in v.indices) out[i] = v[i] / n
        return out
    }

    fun close() = engine.close()
}
