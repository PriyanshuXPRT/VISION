package com.vision.app.ui.screens.users

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.vision.app.data.repository.UserRepository
import com.vision.app.domain.model.User
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

@HiltViewModel
class UserManagementViewModel @Inject constructor(
    private val userRepository: UserRepository,
) : ViewModel() {
    val users = userRepository.observeAll()
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), emptyList())

    fun delete(id: Long) = viewModelScope.launch { userRepository.delete(id) }
}

@Composable
fun UserManagementScreen(onBack: () -> Unit, vm: UserManagementViewModel = hiltViewModel()) {
    val users by vm.users.collectAsState()
    Scaffold(topBar = { TopAppBar(title = { Text("Users") }, navigationIcon = { TextButton(onBack) { Text("Back") } }) }) { pad ->
        LazyColumn(Modifier.fillMaxSize().padding(pad), contentPadding = PaddingValues(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            items(users) { u -> UserRow(u, onDelete = { vm.delete(u.id) }) }
        }
    }
}

@Composable
private fun UserRow(user: User, onDelete: () -> Unit) {
    ElevatedCard {
        Row(Modifier.fillMaxWidth().padding(12.dp), horizontalArrangement = Arrangement.SpaceBetween) {
            Column(Modifier.weight(1f)) {
                Text(user.name, style = MaterialTheme.typography.titleMedium)
                if (user.phone.isNotEmpty()) Text(user.phone, style = MaterialTheme.typography.bodySmall)
                if (user.email.isNotEmpty()) Text(user.email, style = MaterialTheme.typography.bodySmall)
            }
            TextButton(onClick = onDelete) { Text("Delete", color = MaterialTheme.colorScheme.error) }
        }
    }
}
