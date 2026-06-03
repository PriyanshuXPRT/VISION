package com.vision.app.domain.repository

import com.vision.app.core.ml.AuthenticationResult
import com.vision.app.domain.model.User
import kotlinx.coroutines.flow.Flow

interface AuthRepository {
    fun observeRecent(limit: Int = 100): Flow<List<com.vision.app.data.local.entities.AuthLogEntity>>
    suspend fun log(result: AuthenticationResult, userId: Long?, deviceId: String?)
    suspend fun listDevices(): List<com.vision.app.data.local.entities.DeviceEntity>
}

interface UserRepository {
    fun observeAll(): Flow<List<User>>
    suspend fun create(name: String, phone: String, email: String, notes: String): Long
    suspend fun delete(id: Long)
    suspend fun rebuildIndex()
    suspend fun addTemplate(userId: Long, embedding: FloatArray, quality: Float, source: String = "self"): Long
    fun search(embedding: FloatArray, topK: Int = 5): List<com.vision.app.core.ml.UserMatch>
}
