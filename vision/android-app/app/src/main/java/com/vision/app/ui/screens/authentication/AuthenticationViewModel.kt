package com.vision.app.ui.screens.authentication

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.vision.app.core.ml.AuthenticationResult
import com.vision.app.data.local.entities.AuthLogEntity
import com.vision.app.data.repository.AuthRepository
import com.vision.app.data.repository.UserRepository
import com.vision.app.domain.usecase.AuthenticateUseCase
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class AuthUiState(
    val result: AuthenticationResult? = null,
    val status: String = "Looking for a face…",
    val accepted: Boolean = false,
)

@HiltViewModel
class AuthenticationViewModel @Inject constructor(
    private val authenticate: AuthenticateUseCase,
    private val userRepository: UserRepository,
    private val authRepository: AuthRepository,
) : ViewModel() {
    private val _state = MutableStateFlow(AuthUiState())
    val state = _state.asStateFlow()
    private var acceptedLogged = false

    fun onFrame(bitmap: android.graphics.Bitmap, ts: Long) {
        val r = authenticate.authenticate(bitmap, ts) ?: return
        val status = buildString {
            append("live=${"%.2f".format(r.liveness.finalScore)} ")
            append("sim=${"%.2f".format(r.similarity)} ")
            append("blinks=${r.liveness.blinkCount}")
        }
        _state.value = AuthUiState(result = r, status = status)
        if (r.decision == com.vision.app.core.ml.Decision.ACCEPT && !acceptedLogged) {
            acceptedLogged = true
            viewModelScope.launch {
                authRepository.add(
                    AuthLogEntity(
                        userId = r.userId,
                        timestamp = ts,
                        livenessScore = r.liveness.finalScore,
                        similarityScore = r.similarity,
                        result = r.decision.raw,
                        reason = r.reason,
                        deviceId = null,
                        frames = r.framesProcessed,
                        latencyMs = r.latencyMs,
                        spoofKind = r.liveness.spoofKind.raw,
                    )
                )
            }
        }
    }
}
