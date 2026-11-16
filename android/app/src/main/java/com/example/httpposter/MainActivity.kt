package com.example.httpposter

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.view.View
import android.view.inputmethod.InputMethodManager
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import kotlin.concurrent.thread

/**
 * One screen: a server URL, a multiline box for arbitrary text, and the
 * response of the server below it. Text is typed or dictated with the
 * microphone, which is transcribed by the server. An image can be attached
 * and sent together with the prompt. Replies are read out loud unless the
 * Speak box is unchecked.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var serverUrl: EditText
    private lateinit var prompt: EditText
    private lateinit var response: TextView
    private lateinit var sendButton: Button
    private lateinit var voiceButton: Button
    private lateinit var attachButton: Button
    private lateinit var removeImageButton: Button
    private lateinit var imagePreview: ImageView
    private lateinit var imageStatus: TextView
    private lateinit var speakCheck: CheckBox
    private lateinit var speaker: Speaker

    private var timeoutSeconds: Int = ServerConfig.DEFAULT_TIMEOUT_SECONDS
    private var session: String? = null
    private var selectedImageBytes: ByteArray? = null
    private var selectedImageMime: String? = null

    private val pickImage = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            loadSelectedImage(uri)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        serverUrl = findViewById(R.id.serverUrl)
        prompt = findViewById(R.id.prompt)
        response = findViewById(R.id.response)
        sendButton = findViewById(R.id.sendButton)
        voiceButton = findViewById(R.id.voiceButton)
        attachButton = findViewById(R.id.attachButton)
        removeImageButton = findViewById(R.id.removeImageButton)
        imagePreview = findViewById(R.id.imagePreview)
        imageStatus = findViewById(R.id.imageStatus)
        speakCheck = findViewById(R.id.speakCheck)

        speaker = Speaker(this)
        speakCheck.setOnCheckedChangeListener { _, checked ->
            if (!checked) speaker.stop()
        }

        val config = ServerConfig.load(this)
        serverUrl.setText(config.url)
        timeoutSeconds = config.timeoutSeconds

        findViewById<Button>(R.id.saveButton).setOnClickListener {
            val url = serverUrl.text.toString().trim()
            if (url.isEmpty()) {
                toast("Enter a server URL")
                return@setOnClickListener
            }
            ServerConfig.saveUrl(this, url)
            toast("Saved")
        }

        sendButton.setOnClickListener { send() }
        voiceButton.setOnClickListener { onVoiceClick() }
        attachButton.setOnClickListener { pickImage.launch("image/*") }
        removeImageButton.setOnClickListener { clearSelectedImage() }

        findViewById<Button>(R.id.clearButton).setOnClickListener {
            speaker.stop()
            prompt.text.clear()
            response.text = ""
            session = null
            clearSelectedImage()
        }
    }

    private fun send() {
        val url = serverUrl.text.toString().trim()
        val text = prompt.text.toString().trim()
        val imageBytes = selectedImageBytes
        val imageMime = selectedImageMime

        if (url.isEmpty()) {
            toast("Enter a server URL")
            return
        }
        if (text.isEmpty() && imageBytes == null) {
            toast("Type something or attach an image")
            return
        }

        hideKeyboard()
        prompt.clearFocus()
        speaker.stop()
        sendButton.isEnabled = false
        response.text = getString(R.string.thinking)

        thread {
            try {
                val result = if (imageBytes != null && imageMime != null) {
                    HttpPoster.postWithImage(url, text, imageBytes, imageMime, session, timeoutSeconds)
                } else {
                    HttpPoster.post(url, text, session, timeoutSeconds)
                }
                session = result.session
                runOnUiThread {
                    hideKeyboard()
                    response.text = result.reply
                    prompt.text.clear()
                    clearSelectedImage()
                    sendButton.isEnabled = true
                    if (speakCheck.isChecked) speaker.speak(result.reply)
                }
            } catch (e: Exception) {
                runOnUiThread {
                    response.text = "Error: ${e.message ?: e.javaClass.simpleName}"
                    sendButton.isEnabled = true
                }
            }
        }
    }

    private fun hideKeyboard() {
        val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
        imm.hideSoftInputFromWindow(serverUrl.windowToken, 0)
    }

    private fun toast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    private fun onVoiceClick() {
        if (VoiceRecorder.isRecording()) {
            voiceButton.isEnabled = false
            stopAndTranscribe()
        } else if (ContextCompat.checkSelfPermission(
                this, Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            startRecording()
        } else {
            requestPermissions(
                arrayOf(Manifest.permission.RECORD_AUDIO), REQUEST_RECORD_AUDIO
            )
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_RECORD_AUDIO) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startRecording()
            } else {
                toast("The microphone is needed for voice input")
            }
        }
    }

    private fun startRecording() {
        speaker.stop()
        try {
            VoiceRecorder.start()
        } catch (e: Exception) {
            toast("Cannot record: ${e.message ?: e.javaClass.simpleName}")
            return
        }
        voiceButton.text = getString(R.string.stop)
        sendButton.isEnabled = false
        response.text = getString(R.string.listening)
    }

    private fun stopAndTranscribe() {
        val wav = VoiceRecorder.stop()
        voiceButton.text = getString(R.string.voice)
        if (wav.isEmpty()) {
            toast("Nothing was recorded")
            sendButton.isEnabled = true
            return
        }

        val url = transcribeUrl(serverUrl.text.toString().trim())
        response.text = getString(R.string.transcribing)

        thread {
            try {
                val transcript = HttpPoster.transcribe(url, wav, timeoutSeconds)
                runOnUiThread {
                    if (prompt.text.isNotEmpty()) prompt.append(" ")
                    prompt.append(transcript)
                    hideKeyboard()
                    response.text = ""
                    sendButton.isEnabled = true
                    voiceButton.isEnabled = true
                }
            } catch (e: Exception) {
                runOnUiThread {
                    response.text = "Error: ${e.message ?: e.javaClass.simpleName}"
                    sendButton.isEnabled = true
                    voiceButton.isEnabled = true
                }
            }
        }
    }

    private fun transcribeUrl(chatUrl: String): String {
        val base = if (chatUrl.endsWith("/chat")) {
            chatUrl.dropLast("/chat".length)
        } else {
            chatUrl.trimEnd('/')
        }
        return "$base/transcribe"
    }

    private fun loadSelectedImage(uri: Uri) {
        try {
            val mime = contentResolver.getType(uri) ?: "image/jpeg"
            val bytes = contentResolver.openInputStream(uri)?.use { it.readBytes() }
                ?: throw IllegalArgumentException("Cannot read the selected image")
            val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                ?: throw IllegalArgumentException("Unsupported image format")
            selectedImageBytes = bytes
            selectedImageMime = mime
            imagePreview.setImageBitmap(bitmap)
            imagePreview.visibility = View.VISIBLE
            imageStatus.text = getString(R.string.image_attached)
            imageStatus.visibility = View.VISIBLE
            removeImageButton.visibility = View.VISIBLE
        } catch (e: Exception) {
            clearSelectedImage()
            toast("Cannot attach image: ${e.message ?: e.javaClass.simpleName}")
        }
    }

    private fun clearSelectedImage() {
        selectedImageBytes = null
        selectedImageMime = null
        imagePreview.setImageDrawable(null)
        imagePreview.visibility = View.GONE
        imageStatus.visibility = View.GONE
        removeImageButton.visibility = View.GONE
    }

    override fun onDestroy() {
        speaker.shutdown()
        super.onDestroy()
    }

    companion object {
        private const val REQUEST_RECORD_AUDIO = 1
    }
}
