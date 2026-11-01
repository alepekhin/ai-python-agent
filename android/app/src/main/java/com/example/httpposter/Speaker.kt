package com.example.httpposter

import android.content.Context
import android.speech.tts.TextToSpeech
import java.util.Locale

/**
 * Reads replies out loud with the Android text-to-speech engine.
 *
 * The voice is picked from the script of the reply: mostly Cyrillic is spoken
 * with the Russian voice, anything else with the English one. The engine
 * initializes asynchronously, so a reply that arrives too early is held and
 * spoken once it is ready. A device without a usable engine simply stays
 * silent instead of breaking the dialog.
 */
class Speaker(context: Context) {

    private var tts: TextToSpeech? = null
    private var ready = false
    private var failed = false
    private var pending: String? = null

    init {
        tts = TextToSpeech(context.applicationContext) { status ->
            if (status == TextToSpeech.SUCCESS) {
                ready = true
                pending?.let { text ->
                    pending = null
                    say(text)
                }
            } else {
                failed = true
            }
        }
    }

    /**
     * Read the reply out loud, cutting off whatever is being spoken now.
     *
     * The text is cleaned first: code blocks are dropped, links and markup
     * are reduced to words a voice can read, and a long reply is capped.
     */
    fun speak(text: String) {
        if (failed) return
        val russian = isRussian(text)
        val spoken = speakable(text, if (russian) LINK_WORD_RU else LINK_WORD_EN)
        if (spoken.isEmpty()) return
        if (!ready) {
            pending = spoken
            return
        }
        say(spoken)
    }

    /** Stop the current utterance, if any. */
    fun stop() {
        pending = null
        if (!ready) return
        try {
            tts?.stop()
        } catch (e: IllegalStateException) {
            // The engine went away between the check and the call; nothing to do.
        }
    }

    /** Stop speaking and release the engine. */
    fun shutdown() {
        pending = null
        ready = false
        try {
            tts?.stop()
            tts?.shutdown()
        } catch (e: IllegalStateException) {
            // Already released.
        }
        tts = null
    }

    private fun say(text: String) {
        val engine = tts ?: return
        val language = if (isRussian(text)) Locale("ru", "RU") else Locale.US
        val result = engine.setLanguage(language)
        if (result == TextToSpeech.LANG_MISSING_DATA ||
            result == TextToSpeech.LANG_NOT_SUPPORTED
        ) {
            // No data for that voice: say it with the device language instead
            // of not saying it at all.
            engine.setLanguage(Locale.getDefault())
        }
        engine.speak(text, TextToSpeech.QUEUE_FLUSH, null, UTTERANCE_ID)
    }

    private fun isRussian(text: String): Boolean {
        return CYRILLIC.findAll(text).count() > LATIN.findAll(text).count()
    }

    private fun speakable(text: String, linkWord: String): String {
        var spoken = CODE_BLOCK.replace(text, " ")
        spoken = MARKDOWN_LINK.replace(spoken, "\$1")
        spoken = URL.replace(spoken, " $linkWord ")
        spoken = spoken.replace("`", " ")
        spoken = MARKUP.replace(spoken, " ")
        spoken = spoken.replace(Regex("\\s+"), " ").trim()
        if (spoken.length > MAX_CHARS) {
            spoken = spoken.take(MAX_CHARS).substringBeforeLast(" ", spoken) + ", and more."
        }
        return spoken
    }

    companion object {
        private const val MAX_CHARS = 1200
        private const val UTTERANCE_ID = "reply"
        private const val LINK_WORD_EN = "link"
        private const val LINK_WORD_RU = "ссылка"

        private val CODE_BLOCK = Regex("```.*?(?:```|\\z)", RegexOption.DOT_MATCHES_ALL)
        private val MARKDOWN_LINK = Regex("\\[([^\\]]*)]\\([^)]*\\)")
        private val URL = Regex("(?:https?://|www\\.)\\S+")
        private val MARKUP = Regex("[*_#>~|\\[\\]]")
        private val CYRILLIC = Regex("[Ѐ-ӿ]")
        private val LATIN = Regex("[A-Za-z]")
    }
}
