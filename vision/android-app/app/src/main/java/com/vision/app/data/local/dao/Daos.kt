package com.vision.app.data.local.dao

import androidx.room.*
import com.vision.app.data.local.entities.*
import kotlinx.coroutines.flow.Flow

@Dao
interface UserDao {
    @Query("SELECT * FROM users WHERE isActive = 1 ORDER BY createdAt DESC")
    fun observeAll(): Flow<List<UserEntity>>

    @Query("SELECT * FROM users WHERE userId = :id")
    suspend fun get(id: Long): UserEntity?

    @Insert
    suspend fun insert(u: UserEntity): Long

    @Update
    suspend fun update(u: UserEntity)

    @Query("DELETE FROM users WHERE userId = :id")
    suspend fun delete(id: Long)
}

@Dao
interface FaceTemplateDao {
    @Query("SELECT * FROM face_templates ORDER BY userId, isPrimary DESC")
    suspend fun listAll(): List<FaceTemplateEntity>

    @Query("SELECT * FROM face_templates WHERE userId = :userId ORDER BY isPrimary DESC, createdAt DESC")
    suspend fun listForUser(userId: Long): List<FaceTemplateEntity>

    @Insert
    suspend fun insert(t: FaceTemplateEntity): Long

    @Query("DELETE FROM face_templates WHERE templateId = :id")
    suspend fun delete(id: Long)

    @Query("DELETE FROM face_templates WHERE userId = :userId")
    suspend fun deleteForUser(userId: Long)
}

@Dao
interface AuthLogDao {
    @Query("SELECT * FROM auth_logs ORDER BY timestamp DESC LIMIT :limit")
    fun observeRecent(limit: Int = 100): Flow<List<AuthLogEntity>>

    @Insert
    suspend fun insert(log: AuthLogEntity): Long

    @Query("DELETE FROM auth_logs WHERE timestamp < :cutoff")
    suspend fun purgeOlderThan(cutoff: Long): Int
}

@Dao
interface DeviceDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun upsert(d: DeviceEntity)

    @Query("SELECT * FROM devices")
    suspend fun listAll(): List<DeviceEntity>

    @Query("SELECT * FROM devices WHERE deviceId = :id")
    suspend fun get(id: String): DeviceEntity?
}
