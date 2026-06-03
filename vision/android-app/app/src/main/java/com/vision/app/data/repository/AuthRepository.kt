package com.vision.app.data.repository

import com.vision.app.data.local.dao.AuthLogDao
import com.vision.app.data.local.dao.DeviceDao
import com.vision.app.data.local.entities.AuthLogEntity
import com.vision.app.data.local.entities.DeviceEntity
import javax.inject.Inject
import javax.inject.Singleton
import kotlinx.coroutines.flow.Flow

@Singleton
class AuthRepository @Inject constructor(
    private val logDao: AuthLogDao,
    private val deviceDao: DeviceDao,
) {
    fun observeRecent(limit: Int = 100): Flow<List<AuthLogEntity>> = logDao.observeRecent(limit)
    suspend fun add(log: AuthLogEntity): Long = logDao.insert(log)
    suspend fun purgeOlderThan(cutoff: Long) = logDao.purgeOlderThan(cutoff)
    suspend fun registerDevice(d: DeviceEntity) = deviceDao.upsert(d)
    suspend fun listDevices(): List<DeviceEntity> = deviceDao.listAll()
}
