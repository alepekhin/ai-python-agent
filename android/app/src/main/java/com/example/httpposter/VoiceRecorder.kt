package com.example.httpposter

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.concurrent.thread

/**
 * Records the microphone into a mono 16-bit PCM WAV held in memory and gives
 * the bytes back on stop. The recording is 16 kHz, the rate the agent's model
 * transcribes.
 */
object VoiceRecorder {

    const val SAMPLE_RATE = 16_000

    private val lock = Any()

    private class State(val record: AudioRecord) {
        val samples = ByteArrayOutputStream()
        var drain: Thread? = null
    }

    private var state: State? = null

    /** Whether the microphone is open and recording right now. */
    fun isRecording(): Boolean = synchronized(lock) { state != null }

    /** Open the microphone and start collecting samples. */
    fun start() {
        synchronized(lock) {
            if (state != null) return
            val minBuffer = AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            )
            if (minBuffer <= 0) {
                throw IllegalStateException("the microphone is not available")
            }
            val record = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                minBuffer
            )
            if (record.state != AudioRecord.STATE_INITIALIZED) {
                record.release()
                throw IllegalStateException("cannot open the microphone")
            }
            val next = State(record)
            state = next
            record.startRecording()
            next.drain = thread(name = "voice-record") {
                drain(record, next.samples, minBuffer)
            }
        }
    }

    /** Stop recording and return the recording as a WAV file. */
    fun stop(): ByteArray {
        val held = synchronized(lock) {
            val held = state
            state = null
            held
        }
        if (held == null) return byteArrayOf()
        try {
            held.record.stop()
        } catch (e: IllegalStateException) {
            // Not recording after all; the samples are returned anyway.
        }
        held.drain?.join()
        held.record.release()
        val pcm = synchronized(held.samples) { held.samples.toByteArray() }
        return wavFromPcm(pcm)
    }

    private fun drain(record: AudioRecord, samples: ByteArrayOutputStream, minBuffer: Int) {
        val buffer = ByteArray(minBuffer.coerceAtLeast(4096))
        while (synchronized(lock) { state?.record === record }) {
            val read = record.read(buffer, 0, buffer.size)
            if (read <= 0) break
            synchronized(samples) { samples.write(buffer, 0, read) }
        }
    }

    private fun wavFromPcm(pcm: ByteArray): ByteArray {
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN).apply {
            put("RIFF".toByteArray(Charsets.US_ASCII))
            putInt(36 + pcm.size)
            put("WAVE".toByteArray(Charsets.US_ASCII))
            put("fmt ".toByteArray(Charsets.US_ASCII))
            putInt(16) // PCM chunk size
            putShort(1) // PCM format
            putShort(1) // mono
            putInt(SAMPLE_RATE)
            putInt(SAMPLE_RATE * 2) // byte rate
            putShort(2) // block align
            putShort(16) // bits per sample
            put("data".toByteArray(Charsets.US_ASCII))
            putInt(pcm.size)
        }
        return header.array() + pcm
    }
}