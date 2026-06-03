package com.vision.app.ui.screens.admin

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.Group
import androidx.compose.material.icons.filled.History
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.hilt.navigation.compose.hiltViewModel

@Composable
fun AdminDashboardScreen(
    onRegister: () -> Unit,
    onUsers: () -> Unit,
    onLogs: () -> Unit,
    onModels: () -> Unit,
    onSettings: () -> Unit,
    vm: AdminDashboardViewModel = hiltViewModel(),
) {
    Scaffold(topBar = { TopAppBar(title = { Text("Admin Dashboard") }) }) { pad ->
        Column(
            modifier = Modifier.fillMaxSize().padding(pad).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            ElevatedCard {
                Column(Modifier.padding(20.dp)) {
                    Text("Users", fontSize = 14.sp)
                    Text("${vm.userCount}", fontSize = 32.sp)
                }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Tile("Register user", Icons.Default.Face, onRegister, modifier = Modifier.weight(1f))
                Tile("User management", Icons.Default.Group, onUsers, modifier = Modifier.weight(1f))
            }
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Tile("Auth logs", Icons.Default.History, onLogs, modifier = Modifier.weight(1f))
                Tile("Models", Icons.Default.Memory, onModels, modifier = Modifier.weight(1f))
            }
            Tile("Settings", Icons.Default.Settings, onSettings, modifier = Modifier.fillMaxWidth())
        }
    }
}

@Composable
private fun Tile(label: String, icon: androidx.compose.ui.graphics.vector.ImageVector, onClick: () -> Unit, modifier: Modifier = Modifier) {
    ElevatedCard(onClick = onClick, modifier = modifier) {
        Column(Modifier.padding(20.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Icon(icon, contentDescription = null, modifier = Modifier.size(36.dp))
            Spacer(Modifier.height(8.dp))
            Text(label)
        }
    }
}
