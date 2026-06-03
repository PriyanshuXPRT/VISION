package com.vision.app

import com.vision.app.core.ml.embedding_search.BruteForceIndex
import org.junit.Assert.assertEquals
import org.junit.Test

class EmbeddingIndexTest {
    @Test
    fun `empty index returns nothing`() {
        val idx = BruteForceIndex(8)
        assertEquals(0, idx.size)
    }

    @Test
    fun `exact match returns the inserted id`() {
        val idx = BruteForceIndex(4)
        val v = floatArrayOf(1f, 0f, 0f, 0f)
        idx.add(42L, v)
        val hits = idx.search(v, 1)
        assertEquals(1, hits.size)
        assertEquals(42L, hits[0].id)
    }
}
