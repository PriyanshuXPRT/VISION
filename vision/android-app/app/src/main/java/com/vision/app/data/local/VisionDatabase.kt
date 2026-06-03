package com.vision.app.data.local

import androidx.room.Database
import androidx.room.RoomDatabase
import com.vision.app.data.local.dao.AuthLogDao
import com.vision.app.data.local.dao.DeviceDao
import com.vision.app.data.local.dao.FaceTemplateDao
import com.vision.app.data.local.dao.UserDao
import com.vision.app.data.local.entities.AuthLogEntity
import com.vision.app.data.local.entities.DeviceEntity
import com.vision.app.data.local.entities.FaceTemplateEntity
import com.vision.app.data.local.entities.UserEntity

@Database(
    entities = [UserEntity::class, FaceTemplateEntity::class, AuthLogEntity::class, DeviceEntity::class],
    version = 1,
    exportSchema = true,
)
abstract class VisionDatabase : RoomDatabase() {
    abstract fun userDao(): UserDao
    abstract fun templateDao(): FaceTemplateDao
    abstract fun logDao(): AuthLogDao
    abstract fun deviceDao(): DeviceDao
}
