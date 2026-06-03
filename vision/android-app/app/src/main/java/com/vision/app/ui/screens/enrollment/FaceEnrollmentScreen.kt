package com.vision.app.ui.screens.enrollment

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
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.hilt.navigation.compose.hiltViewModel
import androidx.lifecycle.LifecycleOwner

@Composable
fun FaceEnrollmentScreen(
    userId: Long,
    onDone: () -> Unit,
    vm: FaceEnrollmentViewModel = hiltViewModel(),
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

    Scaffold(topBar = { TopAppBar(title = { Text("Face Enrollment") }) }) { pad ->
        Column(Modifier.fillMaxSize().padding(pad).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            Text("Look at the camera, blink naturally, and slowly turn your head side to side. Hold steady for 3 seconds.")
            Box(modifier = Modifier.fillMaxWidth().weight(1f)) {
                if (hasPermission) CameraPreview(lifecycle) { bitmap, ts -> vm.onFrame(userId, bitmap, ts) }
                else Text("Camera permission required.", modifier = Modifier.align(Alignment.Center))
            }
            LinearProgressIndicator(progress = { state.progress }, modifier = Modifier.fillMaxWidth())
            Text(state.status, style = MaterialTheme.typography.bodySmall)
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                OutlinedButton(onClick = onDone, modifier = Modifier.weight(1f)) { Text("Cancel") }
                Button(onClick = { vm.finish(onDone) }, enabled = state.canFinish, modifier = Modifier.weight(1f)) { Text("Done") }
            }
        }
    }
}

@Composable
private fun CameraPreview(
    lifecycle: LifecycleOwner,
    onFrame: (android.graphics.Bitmap, Long) -> Unit,
) {
    AndroidView(factory = { ctx ->
        val preview = PreviewView(ctx)
        val providerFuture = ProcessCameraProvider.getInstance(ctx)
        providerFuture.addListener({
            val provider = providerFuture.get()
            val previewUse = Preview.Builder().build().also { it.setSurfaceProvider(preview.surfaceProvider) }
            val analyzer = androidx.camera.core.ImageAnalysis.Builder()
                .setBackpressureStrategy(androidx.camera.core.ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build().also {
                    it.setAnalyzer(
                        java.util.concurrent.Executors.newSingleThreadExecutor(),
                        com.vision.app.core.camera.RgbImageAnalyzer { bmp, ts -> onFrame(bmp, ts) },
                    )
                }
            provider.unbindAll()
            provider.bindToLifecycle(lifecycle, CameraSelector.DEFAULT_FRONT_CAMERA, previewUse, analyzer)
        }, ContextCompat.getMainExecutor(ctx))
        preview
    }, modifier = Modifier.fillMaxSize())
}
