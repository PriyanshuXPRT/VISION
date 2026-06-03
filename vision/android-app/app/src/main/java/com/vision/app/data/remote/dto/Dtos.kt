package com.vision.app.data.remote.dto

import kotlinx.serialization.Serializable

@Serializable
data class SyncPullRequest(val since: String? = null, val includeTemplates: Boolean = true)

@Serializable
data class SyncPullResponse(
    val serverTime: String,
    val users: List<UserDto>,
    val templates: List<TemplateDto>,
    val cursor: String,
)

@Serializable
data class SyncPushRequest(
    val templates: List<TemplateDto> = emptyList(),
    val logs: List<LogDto> = emptyList(),
)

@Serializable
data class SyncPushResponse(val accepted: Int, val rejected: Int, val serverTime: String)

@Serializable
data class UserDto(
    val userId: Long,
    val name: String,
    val phone: String,
    val email: String,
    val notes: String,
    val isActive: Boolean,
    val createdAt: String,
    val updatedAt: String,
)

@Serializable
data class TemplateDto(
    val userId: Long,
    val embedding: List<Float>,
    val qualityScore: Float,
    val source: String,
    val isPrimary: Boolean,
    val createdAt: String,
)

@Serializable
data class LogDto(
    val userId: Long?,
    val timestamp: String,
    val livenessScore: Float,
    val similarityScore: Float,
    val result: String,
    val reason: String,
    val deviceId: String?,
    val frames: Int,
    val latencyMs: Long,
    val spoofKind: String,
)
