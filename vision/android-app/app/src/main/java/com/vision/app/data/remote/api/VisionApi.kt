package com.vision.app.data.remote.api

import com.vision.app.data.remote.dto.SyncPullRequest
import com.vision.app.data.remote.dto.SyncPullResponse
import com.vision.app.data.remote.dto.SyncPushRequest
import com.vision.app.data.remote.dto.SyncPushResponse
import io.ktor.client.HttpClient
import io.ktor.client.call.body
import io.ktor.client.request.get
import io.ktor.client.request.header
import io.ktor.client.request.post
import io.ktor.client.request.setBody
import io.ktor.client.statement.HttpResponse
import io.ktor.http.ContentType
import io.ktor.http.HttpHeaders
import io.ktor.http.contentType
import javax.inject.Inject
import javax.inject.Singleton

/**
 * Thin Ktor wrapper around the optional hybrid sync endpoints. Used only when
 * the user enables "Hybrid sync" in Settings.
 */
@Singleton
class VisionApi @Inject constructor(private val client: HttpClient) {
    suspend fun pull(req: SyncPullRequest, token: String): SyncPullResponse =
        client.post("v1/sync/pull") {
            header(HttpHeaders.Authorization, "Bearer $token")
            contentType(ContentType.Application.Json)
            setBody(req)
        }.body()

    suspend fun push(req: SyncPushRequest, token: String): SyncPushResponse =
        client.post("v1/sync/push") {
            header(HttpHeaders.Authorization, "Bearer $token")
            contentType(ContentType.Application.Json)
            setBody(req)
        }.body()

    suspend fun health(): HttpResponse =
        client.get("v1/healthz")
}
