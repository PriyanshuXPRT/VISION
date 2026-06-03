package com.vision.app.ui.screens.admin

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.vision.app.data.repository.UserRepository
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

@HiltViewModel
class AdminDashboardViewModel @Inject constructor(
    private val userRepository: UserRepository,
) : ViewModel() {
    private val _userCount = MutableStateFlow(0)
    val userCount: Int get() = _userCount.value

    init {
        viewModelScope.launch {
            userRepository.size.collect { _userCount.value = it }
        }
    }
}
