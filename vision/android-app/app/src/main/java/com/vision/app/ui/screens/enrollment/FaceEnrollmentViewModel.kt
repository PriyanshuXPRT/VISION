package com.vision.app.ui.screens.enrollment

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.vision.app.data.repository.UserRepository
import com.vision.app.domain.usecase.AuthenticateUseCase
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

data class EnrollmentState(
    val progress: Float = 0f,
    val status: String = "Capturing frames…",
    val canFinish: Boolean = false,
)

@HiltViewModel
class FaceEnrollmentViewModel @Inject constructor(
    private val authenticate: AuthenticateUseCase,
    private val userRepository: UserRepository,
) : ViewModel() {
    private val _state = MutableStateFlow(EnrollmentState())
    val state = _state.asStateFlow()
    private var frames = 0
    private val required = 30
    private var saved = false

    fun onFrame(userId: Long, bitmap: android.graphics.Bitmap, ts: Long) {
        val result = authenticate.authenticate(bitmap, ts) ?: return
        frames += 1
        val progress = (frames.toFloat() / required).coerceAtMost(1f)
        val status = "Captured $frames / $required frames · similarity=${"%.2f".format(result.similarity)}"
        val canFinish = frames >= required && result.similarity >= 0.40f && result.liveness.finalScore >= 0.6f
        _state.value = EnrollmentState(progress, status, canFinish)

        if (canFinish && !saved) {
            saved = true
            // Persist a template (uses the latest embedding in the pipeline).
            viewModelScope.launch {
                val embedding = lastEmbedding
                if (embedding != null) {
                    userRepository.addTemplate(userId, embedding, quality = result.liveness.finalScore, source = "self")
                    _state.value = _state.value.copy(status = "Saved face template")
                }
            }
        }
    }

    private var lastEmbedding: FloatArray? = null

    fun finish(onDone: () -> Unit) {
        onDone()
    }
}
