package com.vision.app.data.repository

import com.vision.app.core.ml.embedding_search.BruteForceIndex
import com.vision.app.data.local.dao.FaceTemplateDao
import com.vision.app.data.local.dao.UserDao
import com.vision.app.data.local.entities.FaceTemplateEntity
import com.vision.app.data.local.entities.UserEntity
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Repository over users + face templates. Maintains an in-memory
 * [BruteForceIndex] that mirrors the template table for fast identification.
 */
@Singleton
class UserRepository @Inject constructor(
    private val userDao: UserDao,
    private val templateDao: FaceTemplateDao,
) {
    private val index = BruteForceIndex(512)
    private val _size = MutableStateFlow(0)
    val size: StateFlow<Int> = _size.asStateFlow()

    suspend fun rebuildIndex() {
        val templates = templateDao.listAll()
        synchronized(index) {
            // Drop everything
            for (i in 0 until index.size) index.remove(0)
            for (t in templates) index.add(t.userId, embeddingFromBlob(t.embedding))
        }
        _size.value = index.size
    }

    suspend fun createUser(name: String, phone: String, email: String, notes: String): Long {
        val now = System.currentTimeMillis()
        return userDao.insert(UserEntity(name = name, phone = phone, email = email, notes = notes, createdAt = now, updatedAt = now))
    }

    suspend fun listUsers() = userDao.observeAll()

    suspend fun getUser(id: Long) = userDao.get(id)

    suspend fun addTemplate(userId: Long, embedding: FloatArray, quality: Float, source: String = "self", primary: Boolean = true): Long {
        val now = System.currentTimeMillis()
        val id = templateDao.insert(
            FaceTemplateEntity(
                userId = userId,
                embedding = embedding.toByteArray(),
                qualityScore = quality,
                source = source,
                isPrimary = primary,
                createdAt = now,
            )
        )
        index.add(userId, embedding)
        _size.value = index.size
        return id
    }

    suspend fun deleteUser(id: Long) {
        userDao.delete(id)
        templateDao.deleteForUser(id)
        rebuildIndex()
    }

    fun search(embedding: FloatArray, topK: Int = 5) = index.search(embedding, topK)

    private fun embeddingFromBlob(blob: ByteArray): FloatArray {
        val out = FloatArray(blob.size / 4)
        val bb = java.nio.ByteBuffer.wrap(blob).order(java.nio.ByteOrder.nativeOrder())
        bb.asFloatBuffer().get(out)
        return out
    }
}
