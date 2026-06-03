package com.vision.app.core.ml.embedding_search

import kotlin.math.sqrt

/**
 * Pure-Kotlin vector search for 1:N identification.
 *
 * - For < 1000 users, brute force cosine search is fast enough (sub-10 ms on
 *   mid-range devices) and removes the native dep of FAISS.
 * - Once registered users exceed a threshold, swap in the FAISS-backed
 *   `HierarchicalIndex` (JNI / prebuilt .so).
 */
class BruteForceIndex(private val dim: Int) {
    private val ids = ArrayList<Long>()
    private val matrix = ArrayList<FloatArray>()

    val size: Int get() = ids.size

    @Synchronized
    fun add(id: Long, embedding: FloatArray) {
        require(embedding.size == dim) { "expected dim=$dim, got ${embedding.size}" }
        ids.add(id)
        matrix.add(l2(embedding))
    }

    @Synchronized
    fun remove(id: Long) {
        val idx = ids.indexOf(id)
        if (idx < 0) return
        ids.removeAt(idx); matrix.removeAt(idx)
    }

    data class Hit(val id: Long, val score: Float)

    @Synchronized
    fun search(query: FloatArray, topK: Int): List<Hit> {
        if (matrix.isEmpty()) return emptyList()
        val q = l2(query)
        val scored = matrix.mapIndexed { i, v -> Hit(ids[i], dot(q, v)) }
        return scored.sortedByDescending { it.score }.take(topK)
    }

    private fun l2(v: FloatArray): FloatArray {
        var s = 0f
        for (x in v) s += x * x
        s = sqrt(s.toDouble()).toFloat().coerceAtLeast(1e-12f)
        val out = FloatArray(v.size)
        for (i in v.indices) out[i] = v[i] / s
        return out
    }

    private fun dot(a: FloatArray, b: FloatArray): Float {
        var s = 0f
        for (i in a.indices) s += a[i] * b[i]
        return s
    }
}
