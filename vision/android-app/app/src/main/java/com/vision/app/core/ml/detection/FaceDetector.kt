package com.vision.app.core.ml.detection

import android.content.Context
import com.vision.app.core.ml.BoundingBox
import com.vision.app.core.ml.Detection
import com.vision.app.core.ml.OnnxEngine
import timber.log.Timber
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * SCRFD-10GF detector (Android port).
 *
 * Mirrors `vision.ai.detection.FaceDetector` in Python.
 *
 * Output: at most `maxNum` detections sorted by confidence.
 */
class FaceDetector(
    context: Context,
    modelAsset: String = "models/scrfd_10g_bnkps.onnx",
    private val inputSize: Int = 320,
    private val confThresh: Float = 0.5f,
    private val nmsThresh: Float = 0.4f,
    private val maxNum: Int = 5,
) {
    private val engine = OnnxEngine.fromAsset(context, modelAsset)

    fun detect(rgb: ByteArray, width: Int, height: Int): List<Detection> {
        val (input, scale) = preprocess(rgb, width, height)
        val outs = engine.runFloat(
            input = input.first to input.second,
            shape = input.shape,
        )
        val (bboxes, kpss) = postprocess(outs, scale, width, height)
        if (bboxes.isEmpty()) return emptyList()
        val keep = nms(bboxes, nmsThresh).take(maxNum)
        return keep.map { i ->
            val (x1, y1, x2, y2, s) = bboxes[i]
            val kp = kpss.getOrNull(i) ?: Array(5) { floatArrayOf(0f, 0f) }
            Detection(BoundingBox(x1, y1, x2, y2), s, kp)
        }
    }

    fun detectSingle(rgb: ByteArray, width: Int, height: Int): Detection =
        detect(rgb, width, height).firstOrNull()
            ?: throw IllegalStateException("no_face_detected")

    fun close() = engine.close()

    // --- preprocessing -----------------------------------------------------
    private fun preprocess(rgb: ByteArray, w0: Int, h0: Int): Pair<Triple<String, FloatArray, LongArray>, Float> {
        val det = max(w0, h0)
        val scale = 320f / det
        val nw = (w0 * scale).roundToInt()
        val nh = (h0 * scale).roundToInt()
        val resized = ByteArray(nw * nh * 3)
        // Simplified nearest-neighbour resize to keep the demo small. In production use RenderScript/Bitmap.
        val rIdx = FloatArray(h0 * w0)
        for (i in rIdx.indices) rIdx[i] = i.toFloat()
        // (We use a proper resize helper in production; placeholder is skipped here.)

        val padded = ByteArray(320 * 320 * 3)
        System.arraycopy(resized, 0, padded, 0, resized.size)

        val floats = FloatArray(1 * 3 * 320 * 320)
        for (i in padded.indices) floats[i] = (padded[i].toInt() and 0xFF)
        for (i in floats.indices) floats[i] = (floats[i] - 127.5f) / 128f
        return Triple(engine.inputNames[0], floats, longArrayOf(1, 3, 320, 320)) to scale
    }

    // --- postprocessing ----------------------------------------------------
    private fun postprocess(
        outs: Map<String, Array<FloatArray>>,
        scale: Float,
        w0: Int,
        h0: Int,
    ): Pair<List<FloatArray>, List<Array<FloatArray>>> {
        val bboxes = ArrayList<FloatArray>()
        val kpss = ArrayList<Array<FloatArray>>()
        val fmc = 3
        val anchors = Array(fmc) { i -> anchorCenters(inputSize, inputSize / listOf(8, 16, 32)[i]) }
        for (idx in 0 until fmc) {
            val score = outs["output_${idx}"]?.get(0)?.get(0) ?: continue
            val bbox = outs["output_${idx + fmc}"]?.get(0) ?: continue
            val kps = outs.getOrNull("output_${idx + fmc * 2}")?.get(0)
            val stride = listOf(8, 16, 32)[idx]
            val (h, w) = score.size / (inputSize / stride) to (inputSize / stride)
            for (i in score.indices) {
                if (score[i] < confThresh) continue
                val ax = anchors[idx][i * 2]
                val ay = anchors[idx][i * 2 + 1]
                val (x1, y1, x2, y2) = decodeBbox(bbox, i, ax, ay, scale, stride)
                bboxes.add(floatArrayOf(x1, y1, x2, y2, score[i]))
                k?.let { kpArr ->
                    val pts = Array(5) { p -> floatArrayOf(kpArr[p * 2][i], kpArr[p * 2 + 1][i]) }
                    kpss.add(pts)
                }
            }
        }
        return bboxes to kpss
    }

    private fun anchorCenters(input: Int, stride: Int): FloatArray {
        val n = input / stride
        val out = FloatArray(n * n * 2)
        var k = 0
        for (y in 0 until n) for (x in 0 until n) {
            out[k++] = (x + 0.5f) * stride
            out[k++] = (y + 0.5f) * stride
        }
        return out
    }

    private fun decodeBbox(
        bbox: Array<FloatArray>,
        i: Int,
        ax: Float,
        ay: Float,
        scale: Float,
        stride: Int,
    ): FloatArray {
        val x1 = (ax - bbox[0][i] * 0.5f * stride) / scale
        val y1 = (ay - bbox[1][i] * 0.5f * stride) / scale
        val x2 = (ax + bbox[2][i] * 0.5f * stride) / scale
        val y2 = (ay + bbox[3][i] * 0.5f * stride) / scale
        return floatArrayOf(x1, y1, x2, y2)
    }

    // --- NMS ----------------------------------------------------------------
    private fun nms(dets: List<FloatArray>, iouThresh: Float): List<Int> {
        if (dets.isEmpty()) return emptyList()
        val order = dets.indices.sortedByDescending { dets[it][4] }
        val keep = ArrayList<Int>()
        while (order.isNotEmpty()) {
            val i = order.removeAt(0)
            keep.add(i)
            order.removeAll { j ->
                val iou = iou(dets[i], dets[j])
                iou > iouThresh
            }
        }
        return keep
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val x1 = max(a[0], b[0]); val y1 = max(a[1], b[1])
        val x2 = min(a[2], b[2]); val y2 = min(a[3], b[3])
        val inter = max(0f, x2 - x1) * max(0f, y2 - y1)
        val union = a[2] * a[3] + b[2] * b[3] - inter + 1e-6f
        return inter / union
    }
}
