package com.vision.app.ui.screens.models

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.ViewModel
import dagger.hilt.android.lifecycle.HiltViewModel
import javax.inject.Inject

data class ModelInfo(
    val name: String,
    val version: String,
    val sizeMb: Float,
    val status: String,
    val expectedHash: String? = null,
)

@HiltViewModel
class ModelStatusViewModel @Inject constructor() : ViewModel() {
    val models: List<ModelInfo> = listOf(
        ModelInfo("SCRFD-10GF", "1.0.0", 17.4f, "loaded"),
        ModelInfo("ArcFace R-100", "1.0.0", 248.0f, "loaded"),
        ModelInfo("MiniFASNet 2.7", "1.0.0", 0.6f, "loaded"),
        ModelInfo("MiniFASNet 4.0", "1.0.0", 1.4f, "loaded"),
        ModelInfo("Temporal LSTM", "0.1.0", 0.3f, "not loaded"),
    )
}

@Composable
fun ModelStatusScreen(onBack: () -> Unit, vm: ModelStatusViewModel = hiltViewModel()) {
    Scaffold(topBar = { TopAppBar(title = { Text("Model Status") }, navigationIcon = { TextButton(onBack) { Text("Back") } }) }) { pad ->
        Column(Modifier.fillMaxSize().padding(pad).padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            vm.models.forEach { m ->
                ElevatedCard {
                    Column(Modifier.padding(12.dp)) {
                        Text(m.name, style = MaterialTheme.typography.titleMedium)
                        Text("v${m.version} · ${"%.1f".format(m.sizeMb)} MB", style = MaterialTheme.typography.bodySmall)
                        Text("status: ${m.status}", color = if (m.status == "loaded") MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error)
                    }
                }
            }
        }
    }
}
