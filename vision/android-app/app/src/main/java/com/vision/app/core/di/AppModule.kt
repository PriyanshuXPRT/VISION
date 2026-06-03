package com.vision.app.core.di

import android.content.Context
import androidx.room.Room
import com.vision.app.core.ml.antispoof.AntiSpoof
import com.vision.app.core.ml.detection.FaceDetector
import com.vision.app.core.ml.landmarks.LandmarkEngine
import com.vision.app.core.ml.recognition.FaceRecognizer
import com.vision.app.data.local.VisionDatabase
import com.vision.app.data.local.dao.AuthLogDao
import com.vision.app.data.local.dao.DeviceDao
import com.vision.app.data.local.dao.FaceTemplateDao
import com.vision.app.data.local.dao.UserDao
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import io.ktor.client.HttpClient
import io.ktor.client.engine.okhttp.OkHttp
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.serialization.kotlinx.json.json
import javax.inject.Singleton
import kotlinx.serialization.json.Json

@Module
@InstallIn(SingletonComponent::class)
object AppModule {

    @Provides
    @Singleton
    fun provideDatabase(@ApplicationContext ctx: Context): VisionDatabase =
        Room.databaseBuilder(ctx, VisionDatabase::class.java, "vision.db")
            .fallbackToDestructiveMigration()
            .build()

    @Provides fun userDao(db: VisionDatabase): UserDao = db.userDao()
    @Provides fun templateDao(db: VisionDatabase): FaceTemplateDao = db.templateDao()
    @Provides fun logDao(db: VisionDatabase): AuthLogDao = db.logDao()
    @Provides fun deviceDao(db: VisionDatabase): DeviceDao = db.deviceDao()

    @Provides @Singleton
    fun provideDetector(@ApplicationContext ctx: Context): FaceDetector = FaceDetector(ctx)

    @Provides @Singleton
    fun provideRecognizer(@ApplicationContext ctx: Context): FaceRecognizer = FaceRecognizer(ctx)

    @Provides @Singleton
    fun provideAntispoof(@ApplicationContext ctx: Context): AntiSpoof = AntiSpoof(ctx)

    @Provides @Singleton
    fun provideLandmarks(@ApplicationContext ctx: Context): LandmarkEngine = LandmarkEngine(ctx)

    @Provides @Singleton
    fun provideHttpClient(): HttpClient = HttpClient(OkHttp) {
        install(ContentNegotiation) { json(Json { ignoreUnknownKeys = true }) }
    }
}
