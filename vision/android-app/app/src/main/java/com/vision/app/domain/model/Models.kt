package com.vision.app.domain.model

data class User(
    val id: Long,
    val name: String,
    val phone: String = "",
    val email: String = "",
    val notes: String = "",
    val isActive: Boolean = true,
)

data class AuthLog(
    val id: Long = 0,
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
