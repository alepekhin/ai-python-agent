package com.example.httpposter

import org.json.JSONObject
import java.io.BufferedReader
import java.net.HttpURLConnection
import java.net.URL

/** The reply of the server: the model's answer and the session to continue. */
data class ChatResult(val reply: String, val session: String?)

/**
 * Posts a prompt to the agent server over HTTP.
 *
 * Sends `{"prompt": ...}` as JSON to `/chat` and reads back
 * `{"reply": ..., "session": ...}`. When [session] is known it is sent in the
 * `X-Session-Id` header so the server keeps the conversation history.
 *
 * Blocking: call from a background thread.
 */
object HttpPoster {

    fun post(
        url: String,
        prompt: String,
        session: String?,
        timeoutSeconds: Int
    ): ChatResult {
        val body = JSONObject().put("prompt", prompt).toString()
        val connection = (URL(url).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = timeoutSeconds * 1000
            readTimeout = timeoutSeconds * 1000
            doOutput = true
            setRequestProperty("Content-Type", "application/json; charset=utf-8")
            setRequestProperty("Accept", "application/json")
            if (!session.isNullOrBlank()) {
                setRequestProperty("X-Session-Id", session)
            }
        }

        try {
            connection.outputStream.use { it.write(body.toByteArray(Charsets.UTF_8)) }

            val status = connection.responseCode
            val stream = if (status in 200..299) connection.inputStream else connection.errorStream
            val text = stream?.let { readAll(it.bufferedReader()) } ?: ""

            if (status !in 200..299) {
                val message = parseError(text) ?: "HTTP $status"
                throw RuntimeException(message)
            }

            val json = JSONObject(text)
            val reply = json.optString("reply", "")
            if (reply.isEmpty()) {
                throw RuntimeException("the server did not return a reply")
            }
            val newSession = json.optString("session", "").ifEmpty { null }
            return ChatResult(reply, newSession)
        } finally {
            connection.disconnect()
        }
    }

    private fun readAll(reader: BufferedReader): String {
        return reader.use { it.readText() }
    }

    /**
     * Sends a WAV recording to the server's `/transcribe` and reads back the
     * spoken text: `{"text": ...}`. Blocking: call from a background thread.
     */
    fun transcribe(
        url: String,
        wav: ByteArray,
        timeoutSeconds: Int
    ): String {
        val connection = (URL(url).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = timeoutSeconds * 1000
            readTimeout = timeoutSeconds * 1000
            doOutput = true
            setRequestProperty("Content-Type", "audio/wav")
            setRequestProperty("Accept", "application/json")
        }

        try {
            connection.outputStream.use { it.write(wav) }

            val status = connection.responseCode
            val stream = if (status in 200..299) connection.inputStream else connection.errorStream
            val text = stream?.let { readAll(it.bufferedReader()) } ?: ""

            if (status !in 200..299) {
                val message = parseError(text) ?: "HTTP $status"
                throw RuntimeException(message)
            }

            val json = JSONObject(text)
            val transcript = json.optString("text", "").trim()
            if (transcript.isEmpty()) {
                throw RuntimeException("the server did not return a transcription")
            }
            return transcript
        } finally {
            connection.disconnect()
        }
    }

    private fun parseError(text: String): String? {
        if (text.isBlank()) return null
        return try {
            JSONObject(text).optString("error", "").ifBlank { text }
        } catch (e: Exception) {
            text
        }
    }
}
