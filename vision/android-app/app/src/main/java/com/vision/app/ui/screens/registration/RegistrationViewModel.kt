package com.vision.app.ui.screens.registration

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.vision.app.data.repository.UserRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed interface RegistrationState {
    data object Idle : RegistrationState
    data object Loading : RegistrationState
    data object Success : RegistrationState
    data class Error(val message: String) : RegistrationState
}

@HiltViewModel
class RegistrationViewModel @Inject constructor(
    private val userRepository: UserRepository,
) : ViewModel() {
    private val _state = MutableStateFlow<RegistrationState>(RegistrationState.Idle)
    val state = _state.asStateFlow()

    fun create(name: String, phone: String, email: String, notes: String, onCreated: (Long) -> Unit) {
        _state.value = RegistrationState.Loading
        viewModelScope.launch {
            runCatching { userRepository.createUser(name, phone, email, notes) }
                .onSuccess { uid ->
                    _state.value = RegistrationState.Success
                    onCreated(uid)
                }
                .onFailure { _state.value = RegistrationState.Error(it.message ?: "Failed to create user") }
        }
    }
}
