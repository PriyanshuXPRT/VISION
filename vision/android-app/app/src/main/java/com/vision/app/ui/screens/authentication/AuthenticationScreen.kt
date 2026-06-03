package com.vision.app.ui.screens.authentication

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.hilt.navigation.compose.hiltViewModel
import com.vision.app.core.ml.Decision

@Composable
fun AuthenticationScreen(
    onCancel: () -> Unit,
    vm: AuthenticationViewModel = hiltViewModel(),
) {
    val state by vm.state.collectAsState()
    val ctx = LocalContext.current
    val lifecycle = LocalLifecycleOwner.current
    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(ctx, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }
    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { hasPermission = it }
    LaunchedEffect(Unit) { if (!hasPermission) launcher.launch(Manifest.permission.CAMERA) }

    Scaffold(topBar = { TopAppBar(title = { Text("Authenticate") }) }) { pad ->
        Box(Modifier.fillMaxSize().padding(pad)) {
            if (hasPermission) {
                AndroidView(factory = { c ->
                    val preview = PreviewView(c)
                    val providerFuture = ProcessCameraProvider.getInstance(c)
                    providerFuture.addListener({
                        val provider = providerFuture.get()
                        val p = Preview.Builder().build().also { it.setSurfaceProvider(preview.surfaceProvider) }
                        val analyzer = androidx.camera.core.ImageAnalysis.Builder()
                            .setBackpressureStrategy(androidx.camera.core.ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .build()
                            .also {
                                it.setAnalyzer(
                                    java.util.concurrent.Executors.newSingleThreadExecutor(),
                                    com.vision.app.core.camera.RgbImageAnalyzer { bmp, ts -> vm.onFrame(bmp, ts) },
                                )
                            }
                        provider.unbindAll()
                        provider.bindToLifecycle(lifecycle, CameraSelector.DEFAULT_FRONT_CAMERA, p, analyzer)
                    }, ContextCompat.getMainExecutor(c))
                    preview
                }, modifier = Modifier.fillMaxSize())
            } else {
                Text("Camera permission required", modifier = Modifier.align(Alignment.Center))
            }
            Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.SpaceBetween) {
                Spacer(Modifier.weight(1f))
                val color = when (state.result?.decision) {
                    Decision.ACCEPT -> Color(0xFF2E7D32)
                    Decision.REJECT -> Color(0xFFC62828)
                    else -> Color.Transparent
                }
                Surface(color = color.copy(alpha = 0.7f), modifier = Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp)) {
                        Text(
                            text = state.result?.userName ?: "Authenticating…",
                            color = Color.White,
                            style = MaterialTheme.typography.headlineSmall,
                        )
                        Text(
                            text = state.status,
                            color = Color.White,
                            style = MaterialTheme.typography.bodyMedium,
                        )
                    }
                }
                TextButton(onClick = onCancel) { Text("Cancel") }
            }
        }
    }
}
