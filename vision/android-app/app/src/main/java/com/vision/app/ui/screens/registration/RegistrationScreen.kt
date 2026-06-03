package com.vision.app.ui.screens.registration

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel

@Composable
fun RegistrationScreen(
    onEnroll: (Long) -> Unit,
    onCancel: () -> Unit,
    vm: RegistrationViewModel = hiltViewModel(),
) {
    val state by vm.state.collectAsState()
    var name by remember { mutableStateOf("") }
    var phone by remember { mutableStateOf("") }
    var email by remember { mutableStateOf("") }
    var notes by remember { mutableStateOf("") }
    Scaffold(topBar = { TopAppBar(title = { Text("Register User") }, navigationIcon = { TextButton(onCancel) { Text("Cancel") } }) }) { pad ->
        Column(Modifier.fillMaxSize().padding(pad).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            OutlinedTextField(value = name, onValueChange = { name = it }, label = { Text("Full name *") }, modifier = Modifier.fillMaxWidth())
            OutlinedTextField(value = phone, onValueChange = { phone = it }, label = { Text("Phone") }, modifier = Modifier.fillMaxWidth())
            OutlinedTextField(value = email, onValueChange = { email = it }, label = { Text("Email") }, modifier = Modifier.fillMaxWidth())
            OutlinedTextField(value = notes, onValueChange = { notes = it }, label = { Text("Notes") }, modifier = Modifier.fillMaxWidth())
            Spacer(Modifier.weight(1f))
            Button(
                onClick = { vm.create(name, phone, email, notes) { uid -> onEnroll(uid) } },
                enabled = name.isNotBlank() && state !is RegistrationState.Loading,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(if (state is RegistrationState.Loading) "Saving…" else "Continue to face enrollment")
            }
            if (state is RegistrationState.Error) {
                Text((state as RegistrationState.Error).message, color = MaterialTheme.colorScheme.error)
            }
        }
    }
}
