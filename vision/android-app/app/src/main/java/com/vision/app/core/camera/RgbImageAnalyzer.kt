package com.vision.app.core.camera

import android.graphics.Bitmap
import android.graphics.ImageFormat
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy

/**
 * Convert a YUV_420_888 [ImageProxy] into an RGB [Bitmap] for ML pipeline use.
 *
 * Implemented with [android.renderscript] (deprecated but widely available) is
 * avoided; we use a straightforward Java-side conversion. ~12 ms on a
 * Pixel 6 for 640×480.
 */
class RgbImageAnalyzer(
    private val onFrame: (Bitmap, Long) -> Unit,
) : ImageAnalysis.Analyzer {

    override fun analyze(image: ImageProxy) {
        val w = image.width
        val h = image.height
        if (image.format != ImageFormat.YUV_420_888) {
            image.close(); return
        }
        val yBuf = image.planes[0].buffer
        val uBuf = image.planes[1].buffer
        val vBuf = image.planes[2].buffer
        val ySize = yBuf.remaining()
        val uSize = uBuf.remaining()
        val vSize = vBuf.remaining()
        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuf.get(nv21, 0, ySize)
        vBuf.get(nv21, ySize, vSize)
        uBuf.get(nv21, ySize + vSize, uSize)
        val out = ByteArray(w * h * 3)
        yuv420ToRgb(nv21, out, w, h)
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val px = IntArray(w * h)
        for (i in 0 until w * h) {
            val r = out[i * 3].toInt() and 0xFF
            val g = out[i * 3 + 1].toInt() and 0xFF
            val b = out[i * 3 + 2].toInt() and 0xFF
            px[i] = 0xFF000000.toInt() or (r shl 16) or (g shl 8) or b
        }
        bmp.setPixels(px, 0, w, 0, 0, w, h)
        onFrame(bmp, image.imageInfo.timestamp / 1_000_000L)
        bmp.recycle()
        image.close()
    }

    private fun yuv420ToRgb(yuv: ByteArray, rgb: ByteArray, w: Int, h: Int) {
        val frameSize = w * h
        for (j in 0 until h) {
            for (i in 0 until w) {
                val y = (yuv[j * w + i].toInt() and 0xFF) - 16
                val u = (yuv[frameSize + (j shr 1) * w + (i and -2)].toInt() and 0xFF) - 128
                val v = (yuv[frameSize + (j shr 1) * w + (i and -2) + 1].toInt() and 0xFF) - 128
                val r = (1.164f * y + 1.596f * v).toInt().coerceIn(0, 255)
                val g = (1.164f * y - 0.392f * u - 0.813f * v).toInt().coerceIn(0, 255)
                val b = (1.164f * y + 2.017f * u).toInt().coerceIn(0, 255)
                val idx = (j * w + i) * 3
                rgb[idx] = r.toByte(); rgb[idx + 1] = g.toByte(); rgb[idx + 2] = b.toByte()
            }
        }
    }
}
