package com.vision.app.core.ml

import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import timber.log.Timber
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer

/**
 * Lightweight wrapper around an ONNX Runtime inference session.
 *
 * - Loads the model from `assets/models/<name>.onnx` (or a custom file).
 * - Caches sessions in a process-wide map.
 * - Provides typed helpers for float input tensors.
 */
class OnnxEngine private constructor(
    private val session: OrtSession,
    private val env: OrtEnvironment,
) : AutoCloseable {

    val inputNames: List<String> = session.inputNames.toList()
    val outputNames: List<String> = session.outputNames.toList()

    fun runFloat(
        input: Pair<String, FloatArray>,
        shape: LongArray,
    ): Map<String, Array<FloatArray>> {
        val buffer = FloatBuffer.wrap(input.second)
        val tensor = OnnxTensor.createTensor(env, buffer, shape, OnnxJavaType.FLOAT)
        session.run(mapOf(input.first to tensor)).use { results ->
            val out = LinkedHashMap<String, Array<FloatArray>>()
            for ((idx, name) in outputNames.withIndex()) {
                @Suppress("UNCHECKED_CAST")
                out[name] = (results[idx].value as Array<FloatArray>)
            }
            return out
        }
    }

    fun runFloatBatch(
        input: Pair<String, Array<FloatArray>>,
        shape: LongArray,
    ): Map<String, Array<FloatArray>> {
        val buffer = FloatBuffer.wrap(input.first)
        // Build a proper (B, ...) tensor from a 2D array.
        val tensor = OnnxTensor.createTensor(env, buffer, shape, OnnxJavaType.FLOAT)
        session.run(mapOf(input.first to tensor)).use { results ->
            val out = LinkedHashMap<String, Array<FloatArray>>()
            for ((idx, name) in outputNames.withIndex()) {
                @Suppress("UNCHECKED_CAST")
                out[name] = (results[idx].value as Array<FloatArray>)
            }
            return out
        }
    }

    override fun close() {
        runCatching { session.close() }
    }

    companion object {
        private val cache = HashMap<String, OnnxEngine>()
        private val lock = Any()

        fun fromAsset(
            context: Context,
            assetPath: String,
            intraOpThreads: Int = 2,
        ): OnnxEngine = synchronized(lock) {
            cache[assetPath]?.let { return it }
            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(intraOpThreads)
                setExecutionMode(OrtSession.ExecutionMode.SEQUENTIAL)
                if (android.os.Build.VERSION.SDK_INT >= 29) {
                    addNnapi()  // NNAPI acceleration on Android 10+
                }
            }
            val modelBytes = readAsset(context, assetPath)
            val session = env.createSession(modelBytes, opts)
            Timber.i("ONNX loaded: $assetPath inputs=$inputNames")
            OnnxEngine(session, env).also { cache[assetPath] = it }
        }

        fun fromFile(file: File, intraOpThreads: Int = 2): OnnxEngine = synchronized(lock) {
            val key = file.absolutePath
            cache[key]?.let { return it }
            val env = OrtEnvironment.getEnvironment()
            val opts = OrtSession.SessionOptions().apply {
                setIntraOpNumThreads(intraOpThreads)
                if (android.os.Build.VERSION.SDK_INT >= 29) addNnapi()
            }
            val session = env.createSession(file.absolutePath, opts)
            OnnxEngine(session, env).also { cache[key] = it }
        }

        private fun readAsset(context: Context, assetPath: String): ByteArray {
            val outFile = File(context.cacheDir, assetPath.substringAfterLast('/'))
            if (outFile.exists() && outFile.length() > 0) return outFile.readBytes()
            context.assets.open(assetPath).use { input ->
                FileOutputStream(outFile).use { output -> input.copyTo(output) }
            }
            return outFile.readBytes()
        }
    }
}
