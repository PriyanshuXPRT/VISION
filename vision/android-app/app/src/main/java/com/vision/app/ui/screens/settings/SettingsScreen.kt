package com.vision.app.ui.screens.settings

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.ViewModel
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject

data class AppSettings(
    val hybridSync: Boolean = false,
    val backendUrl: String = "https://api.vision.local",
    val requireBlink: Boolean = true,
    val requireMotion: Boolean = true,
    val livenessThreshold: Float = 0.85f,
    val similarityThreshold: Float = 0.45f,
)

@HiltViewModel
class SettingsViewModel @Inject constructor() : ViewModel() {
    var settings by mutableStateOf(AppSettings())
        private set

    fun update(block: (AppSettings) -> AppSettings) { settings = block(settings) }
}

@Composable
fun SettingsScreen(onBack: () -> Unit, vm: SettingsViewModel = hiltViewModel()) {
    val s = vm.settings
    Scaffold(topBar = { TopAppBar(title = { Text("Settings") }, navigationIcon = { TextButton(onBack) { Text("Back") } }) }) { pad ->
        Column(Modifier.fillMaxSize().padding(pad).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
                Switch(checked = s.hybridSync, onCheckedChange = { v -> vm.update { it.copy(hybridSync = v) } })
                Spacer(Modifier.width(8.dp))
                Text("Enable hybrid sync (encrypted)")
            }
            OutlinedTextField(value = s.backendUrl, onValueChange = { v -> vm.update { it.copy(backendUrl = v) } }, label = { Text("Backend URL") }, modifier = Modifier.fillMaxWidth())
            HorizontalDivider()
            Text("Liveness", style = MaterialTheme.typography.titleMedium)
            Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
                Switch(checked = s.requireBlink, onCheckedChange = { v -> vm.update { it.copy(requireBlink = v) } })
                Spacer(Modifier.width(8.dp)); Text("Require blink")
            }
            Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
                Switch(checked = s.requireMotion, onCheckedChange = { v -> vm.update { it.copy(requireMotion = v) } })
                Spacer(Modifier.width(8.dp)); Text("Require eye motion")
            }
            Text("Liveness threshold: ${"%.2f".format(s.livenessThreshold)}")
            Slider(value = s.livenessThreshold, onValueChange = { v -> vm.update { it.copy(livenessThreshold = v) } }, valueRange = 0.5f..0.99f)
            Text("Similarity threshold: ${"%.2f".format(s.similarityThreshold)}")
            Slider(value = s.similarityThreshold, onValueChange = { v -> vm.update { it.copy(similarityThreshold = v) } }, valueRange = 0.30f..0.80f)
        }
    }
}
