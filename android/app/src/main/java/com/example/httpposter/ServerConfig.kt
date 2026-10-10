package com.example.httpposter

import android.content.Context
import org.json.JSONObject

/**
 * The server the app posts to.
 *
 * The default comes from the bundled `assets/server.json` file; a URL saved
 * from the UI (in SharedPreferences) overrides it, so the app can be pointed
 * at another host without rebuilding.
 */
data class ServerConfig(val url: String, val timeoutSeconds: Int) {

    companion object {
        const val DEFAULT_URL = "http://10.0.2.2:8765/chat"
        const val DEFAULT_TIMEOUT_SECONDS = 60

        private const val PREFS = "http_poster"
        private const val KEY_URL = "server_url"
        private const val ASSET_NAME = "server.json"

        /** The saved URL if the user set one, otherwise the bundled default. */
        fun load(context: Context): ServerConfig {
            val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            val saved = prefs.getString(KEY_URL, null)
            if (!saved.isNullOrBlank()) {
                return ServerConfig(saved, timeoutFromAsset(context))
            }
            return fromAsset(context)
        }

        /** Persist the URL the user typed. */
        fun saveUrl(context: Context, url: String) {
            context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
                .edit()
                .putString(KEY_URL, url.trim())
                .apply()
        }

        private fun fromAsset(context: Context): ServerConfig {
            val json = try {
                context.assets.open(ASSET_NAME).bufferedReader().use { it.readText() }
            } catch (e: Exception) {
                return ServerConfig(DEFAULT_URL, DEFAULT_TIMEOUT_SECONDS)
            }
            return try {
                val obj = JSONObject(json)
                ServerConfig(
                    obj.optString("url", DEFAULT_URL),
                    obj.optInt("timeoutSeconds", DEFAULT_TIMEOUT_SECONDS)
                )
            } catch (e: Exception) {
                ServerConfig(DEFAULT_URL, DEFAULT_TIMEOUT_SECONDS)
            }
        }

        private fun timeoutFromAsset(context: Context): Int {
            return fromAsset(context).timeoutSeconds
        }
    }
}
