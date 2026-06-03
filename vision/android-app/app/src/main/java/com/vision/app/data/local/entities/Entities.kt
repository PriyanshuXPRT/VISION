package com.vision.app.data.local.entities

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "users")
data class UserEntity(
    @PrimaryKey(autoGenerate = true) val userId: Long = 0,
    val name: String,
    val phone: String = "",
    val email: String = "",
    val notes: String = "",
    val isActive: Boolean = true,
    val createdAt: Long,
    val updatedAt: Long,
)

@Entity(tableName = "face_templates")
data class FaceTemplateEntity(
    @PrimaryKey(autoGenerate = true) val templateId: Long = 0,
    val userId: Long,
    val embedding: ByteArray,
    val qualityScore: Float,
    val source: String = "self",
    val isPrimary: Boolean = true,
    val createdAt: Long,
) {
    override fun equals(other: Any?): Boolean = this === other
    override fun hashCode(): Int = System.identityHashCode(this)
}

@Entity(tableName = "auth_logs")
data class AuthLogEntity(
    @PrimaryKey(autoGenerate = true) val logId: Long = 0,
    val userId: Long?,
    val timestamp: Long,
    val livenessScore: Float,
    val similarityScore: Float,
    val result: String,
    val reason: String,
    val deviceId: String?,
    val frames: Int,
    val latencyMs: Long,
    val spoofKind: String,
)

@Entity(tableName = "devices")
data class DeviceEntity(
    @PrimaryKey val deviceId: String,
    val deviceName: String,
    val platform: String,
    val pubkey: ByteArray?,
    val registeredAt: Long,
    val lastSeenAt: Long?,
    val isRevoked: Boolean = false,
) {
    override fun equals(other: Any?): Boolean = this === other
    override fun hashCode(): Int = System.identityHashCode(this)
}
