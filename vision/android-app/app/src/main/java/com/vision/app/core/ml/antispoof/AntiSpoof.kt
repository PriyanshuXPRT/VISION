package com.vision.app.core.ml.antispoof

import android.content.Context
import com.vision.app.core.ml.BoundingBox
import com.vision.app.core.ml.LivenessPrediction
import com.vision.app.core.ml.OnnxEngine
import com.vision.app.core.ml.SpoofKind
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * MiniFASNetV2 ensemble (Android port).
 *
 * Two scale models (2.7 and 4.0) cropped from the same bbox; geometric mean
 * of the per-scale real probabilities is used as the final live score.
 */
class AntiSpoof(context: Context) {
    private val model27 = OnnxEngine.fromAsset(context, "models/minifasnet_2_7.onnx")
    private val model40 = OnnxEngine.fromAsset(context, "models/minifasnet_4_0.onnx")

    fun predict(rgb: ByteArray, width: Int, height: Int, bbox: BoundingBox): LivenessPrediction {
        val (s27, _) = runOne(model27, crop(rgb, width, height, bbox, 2.7f))
        val (s40, _) = runOne(model40, crop(rgb, width, height, bbox, 4.0f))
        val final = sqrt(max(s27, 1e-6f) * max(s40, 1e-6f))
        val isReal = final >= 0.85f
        return LivenessPrediction(final, isReal, if (isReal) SpoofKind.NONE else classifyKind(s27, s40))
    }

    private fun runOne(engine: OnnxEngine, img80: ByteArray): Pair<Float, Float> {
        val floats = FloatArray(3 * 80 * 80)
        for (i in img80.indices) floats[i] = ((img80[i].toInt() and 0xFF) - 127.5f) / 128f
        val outs = engine.runFloat(engine.inputNames[0] to floats, longArrayOf(1, 3, 80, 80))
        val probs = softmax(outs.values.first().first())
        return probs[0] to probs[1]   // (real, fake)
    }

    private fun crop(rgb: ByteArray, w: Int, h: Int, bbox: BoundingBox, scale: Float): ByteArray {
        val cx = (bbox.x1 + bbox.x2) / 2f
        val cy = (bbox.y1 + bbox.y2) / 2f
        val bw = (bbox.x2 - bbox.x1) * scale
        val bh = (bbox.y2 - bbox.y1) * scale
        val x1 = max(0, (cx - bw / 2f).toInt())
        val y1 = max(0, (cy - bh / 2f).toInt())
        val x2 = min(w, (cx + bw / 2f).toInt())
        val y2 = min(h, (cy + bh / 2f).toInt())
        val dst = ByteArray(80 * 80 * 3)
        for (yy in 0 until 80) {
            val sy = y1 + ((y2 - y1) * yy / 80)
            for (xx in 0 until 80) {
                val sx = x1 + ((x2 - x1) * xx / 80)
                val src = (sy * w + sx) * 3
                val d = (yy * 80 + xx) * 3
                dst[d] = rgb[src]
                dst[d + 1] = rgb[src + 1]
                dst[d + 2] = rgb[src + 2]
            }
        }
        return dst
    }

    private fun softmax(v: FloatArray): FloatArray {
        var m = v.max()
        var sum = 0f
        val out = FloatArray(v.size)
        for (i in v.indices) { out[i] = exp((v[i] - m).toDouble()).toFloat(); sum += out[i] }
        for (i in out.indices) out[i] /= sum
        return out
    }

    private fun classifyKind(s27: Float, s40: Float): SpoofKind {
        if (s40 < s27 * 0.6f) return SpoofKind.MOBILE_SCREEN
        if (s27 < 0.1f && s40 < 0.1f) return SpoofKind.PRINT
        if (kotlin.math.abs(s27 - s40) < 0.05f && (s27 + s40) < 0.3f) return SpoofKind.LAPTOP_SCREEN
        if (s27 < s40) return SpoofKind.REPLAY_VIDEO
        return SpoofKind.UNKNOWN
    }

    fun close() {
        model27.close()
        model40.close()
    }
}
