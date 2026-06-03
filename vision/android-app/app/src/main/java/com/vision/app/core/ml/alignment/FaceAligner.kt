package com.vision.app.core.ml.alignment

import kotlin.math.sqrt

/**
 * 5-landmark similarity transform (ArcFace template) — used to align the
 * detected face to a canonical pose before embedding extraction.
 */
object FaceAligner {

    private val ARC_REF_112 = arrayOf(
        floatArrayOf(38.2946f, 51.6963f),
        floatArrayOf(73.5318f, 51.5014f),
        floatArrayOf(56.0252f, 71.7366f),
        floatArrayOf(41.5493f, 92.3655f),
        floatArrayOf(70.7299f, 92.2041f),
    )

    fun warp(
        bitmap: android.graphics.Bitmap,
        landmarks: Array<FloatArray>,
        outputSize: Int = 112,
    ): android.graphics.Bitmap {
        val src = FloatArray(10)
        for (i in 0 until 5) {
            src[i * 2] = landmarks[i][0]
            src[i * 2 + 1] = landmarks[i][1]
        }
        val dst = FloatArray(10)
        val scale = outputSize / 112f
        for (i in 0 until 5) {
            dst[i * 2] = ARC_REF_112[i][0] * scale
            dst[i * 2 + 1] = ARC_REF_112[i][1] * scale
        }
        val m = estimateSimilarity(src, dst)
        val matrix = android.graphics.Matrix()
        matrix.setValues(m)
        return android.graphics.Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun estimateSimilarity(src: FloatArray, dst: FloatArray): FloatArray {
        require(src.size == 10 && dst.size == 10)
        var scx = 0f; var scy = 0f; var dcx = 0f; var dcy = 0f
        for (i in 0 until 5) { scx += src[i*2]; scy += src[i*2+1]; dcx += dst[i*2]; dcy += dst[i*2+1] }
        scx /= 5; scy /= 5; dcx /= 5; dcy /= 5
        var a = 0f; var b = 0f
        for (i in 0 until 5) {
            a += (src[i*2]-scx)*(dst[i*2]-dcx) + (src[i*2+1]-scy)*(dst[i*2+1]-dcy)
            b += (src[i*2]-scx)*(dst[i*2+1]-dcy) - (src[i*2+1]-scy)*(dst[i*2]-dcx)
        }
        val det = sqrt(a*a + b*b) + 1e-6f
        val cos = a / det; val sin = b / det
        return floatArrayOf(
            cos, sin, dcx - scx*cos - scy*sin,
            -sin, cos, dcy - scx*(-sin) - scy*cos,
            0f, 0f, 1f,
        )
    }
}
