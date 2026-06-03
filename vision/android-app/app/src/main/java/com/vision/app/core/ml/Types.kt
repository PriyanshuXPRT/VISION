package com.vision.app.core.ml

/**
 * Domain-level data structures for the Android AI pipeline.
 * Mirror the Python `vision.core.types` module.
 */
package com.vision.app.core.ml

import kotlin.math.max
import kotlin.math.min

data class BoundingBox(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
) {
    val width: Float get() = max(0f, x2 - x1)
    val height: Float get() = max(0f, y2 - y1)
    val area: Float get() = width * height
}

data class Detection(
    val bbox: BoundingBox,
    val score: Float,
    /** Five ArcFace-style landmark points in image pixel coords: (x, y) per point. */
    val landmarks: Array<FloatArray>,
) {
    override fun equals(other: Any?): Boolean = this === other
    override fun hashCode(): Int = System.identityHashCode(this)
}

data class LivenessPrediction(
    val score: Float,
    val isReal: Boolean,
    val spoofKind: SpoofKind,
)

enum class SpoofKind(val raw: String) {
    NONE("none"),
    PRINT("print"),
    MOBILE_SCREEN("mobile_screen"),
    LAPTOP_SCREEN("laptop_screen"),
    REPLAY_VIDEO("replay_video"),
    FACE_SWAP("face_swap"),
    DEEPFAKE("deepfake"),
    MASK("mask"),
    UNKNOWN("unknown");

    companion object {
        fun fromRaw(raw: String?): SpoofKind = values().firstOrNull { it.raw == raw } ?: UNKNOWN
    }
}

enum class LivenessSignal { PASSIVE_FAS, BLINK, EYE_MOTION, HEAD_POSE, TEMPORAL }

enum class Decision(val raw: String) { ACCEPT("accept"), REJECT("reject"), INCONCLUSIVE("inconclusive") }

data class FrameAnalysis(
    val timestampMs: Long,
    val bbox: BoundingBox,
    val alignedFace: ByteArray,                 // 112x112x3 RGB
    val width: Int,
    val height: Int,
    val passiveLiveness: Float,
    val isSpoof: Boolean,
    val spoofKind: SpoofKind,
    val ear: Float,
    val eyeMotion: Float,
    val yaw: Float,
    val pitch: Float,
    val roll: Float,
    val qualityScore: Float,
    val embedding: FloatArray? = null,
) {
    override fun equals(other: Any?): Boolean = this === other
    override fun hashCode(): Int = System.identityHashCode(this)
}

data class LivenessReport(
    val blinkCount: Int = 0,
    val blinkScore: Float = 0f,
    val eyeMotionScore: Float = 0f,
    val headPoseScore: Float = 0f,
    val passiveScore: Float = 0f,
    val temporalScore: Float = 0f,
    val finalScore: Float = 0f,
    val spoofKind: SpoofKind = SpoofKind.NONE,
    val acceptedSignals: List<LivenessSignal> = emptyList(),
    val rejectedSignals: List<LivenessSignal> = emptyList(),
)

data class UserMatch(
    val userId: Long,
    val name: String,
    val similarity: Float,
    val templateId: Long,
)

data class IdentificationResult(
    val bestMatch: UserMatch? = null,
    val candidates: List<UserMatch> = emptyList(),
    val isIdentified: Boolean = false,
)

data class AuthenticationResult(
    val decision: Decision,
    val userId: Long? = null,
    val userName: String? = null,
    val similarity: Float = 0f,
    val liveness: LivenessReport = LivenessReport(),
    val framesProcessed: Int = 0,
    val latencyMs: Long = 0,
    val reason: String = "",
)

fun Float.clip(min: Float = 0f, max: Float = 1f): Float = max(min(min, this), min)
