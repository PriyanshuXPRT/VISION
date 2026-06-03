# Default ProGuard rules for V.I.S.I.O.N. Android app.

-keep class com.vision.app.** { *; }

# ONNX Runtime
-keep class ai.onnxruntime.** { *; }
-dontwarn ai.onnxruntime.**

# CameraX
-keep class androidx.camera.** { *; }
-dontwarn androidx.camera.**

# ML Kit
-keep class com.google.mlkit.** { *; }
-keep class com.google.android.gms.vision.** { *; }

# Room
-keep class androidx.room.** { *; }

# Hilt
-keep class dagger.hilt.** { *; }
-keep class * extends dagger.hilt.android.HiltAndroidApp
-keepclassmembers class ** { @dagger.hilt.android.lifecycle.HiltViewModel <init>(...); }

# kotlinx.serialization
-keepclassmembers class **$$serializer { *; }
-keepattributes *Annotation*, InnerClasses
