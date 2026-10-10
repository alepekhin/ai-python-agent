package com.example.httpposter

import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import kotlin.concurrent.thread

/**
 * One screen: a server URL, a multiline box for arbitrary text, and the
 * response of the server below it.
 */
class MainActivity : AppCompatActivity() {

    private lateinit var serverUrl: EditText
    private lateinit var prompt: EditText
    private lateinit var response: TextView
    private lateinit var sendButton: Button

    private var timeoutSeconds: Int = ServerConfig.DEFAULT_TIMEOUT_SECONDS
    private var session: String? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        serverUrl = findViewById(R.id.serverUrl)
        prompt = findViewById(R.id.prompt)
        response = findViewById(R.id.response)
        sendButton = findViewById(R.id.sendButton)

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

        findViewById<Button>(R.id.clearButton).setOnClickListener {
            prompt.text.clear()
            response.text = ""
            session = null
        }
    }

    private fun send() {
        val url = serverUrl.text.toString().trim()
        val text = prompt.text.toString().trim()

        if (url.isEmpty()) {
            toast("Enter a server URL")
            return
        }
        if (text.isEmpty()) {
            toast("Type something to send")
            return
        }

        sendButton.isEnabled = false
        response.text = getString(R.string.thinking)

        thread {
            try {
                val result = HttpPoster.post(url, text, session, timeoutSeconds)
                session = result.session
                runOnUiThread {
                    response.text = result.reply
                    prompt.text.clear()
                    sendButton.isEnabled = true
                }
            } catch (e: Exception) {
                runOnUiThread {
                    response.text = "Error: ${e.message ?: e.javaClass.simpleName}"
                    sendButton.isEnabled = true
                }
            }
        }
    }

    private fun toast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }
}
