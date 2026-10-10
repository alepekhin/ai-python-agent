# HTTP Poster (Android)

A small Android client for the [AI agent](../README.md) web server. It sends
arbitrary text as an HTTP POST and shows the reply, so the agent can be used
from a phone instead of the terminal.

The app posts `{"prompt": "..."}` as JSON to the server's `/chat` endpoint and
reads back `{"reply": "...", "session": "..."}`. The returned session id is
sent back on the next request in the `X-Session-Id` header, so the server keeps
the conversation history — every message continues the same dialogue until you
press **Clear**.

- One screen: a server URL, a multiline box for the text, and the response.
- The server URL comes from the bundled `assets/server.json` and can be changed
  in the app (saved, and used from then on).
- Plain HTTP (cleartext) is enabled, since the agent serves over `http://`.
- `minSdk 24`, `targetSdk 34` — installs on Android 14.

## Project layout

```
app/src/main/
├── assets/server.json                    # server config file (url, timeoutSeconds)
├── java/com/example/httpposter/
│   ├── MainActivity.kt                   # UI: URL field, prompt, Send/Clear, response
│   ├── ServerConfig.kt                   # loads server.json, saves the URL override
│   └── HttpPoster.kt                     # the POST request and the reply/session parsing
├── res/layout/activity_main.xml
└── AndroidManifest.xml                   # INTERNET permission, cleartext HTTP
```

## Requirements

Everything is done from the terminal — no IDE is needed.

- **JDK 17** — the Android Gradle Plugin and Gradle 8.9 do not support newer
  JDKs; point `JAVA_HOME` at it.
- **Android SDK** — platform 34, build-tools 34.0.0, and platform-tools for
  `adb`.
- The **Gradle wrapper** (`./gradlew`) is committed, so no global Gradle install
  is needed.

### Set up the SDK

If you do not have a JDK 17 and an Android SDK yet, install them into your home
directory (no root required):

```sh
# JDK 17 (Temurin) into ~/jdk-17
mkdir -p ~/jdk-17
curl -sL https://api.adoptium.net/v3/binary/latest/17/ga/linux/x64/jdk/hotspot/normal/eclipse \
  | tar xz -C ~/jdk-17 --strip-components=1

# Android command-line tools into ~/Android/Sdk
mkdir -p ~/Android/Sdk/cmdline-tools
curl -sL https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip -o /tmp/cmdline-tools.zip
unzip -q /tmp/cmdline-tools.zip -d /tmp/cmdline-tools
mv /tmp/cmdline-tools/cmdline-tools ~/Android/Sdk/cmdline-tools/latest

export ANDROID_SDK_ROOT=$HOME/Android/Sdk
yes | $ANDROID_SDK_ROOT/cmdline-tools/latest/bin/sdkmanager --licenses
$ANDROID_SDK_ROOT/cmdline-tools/latest/bin/sdkmanager \
  "platform-tools" "platforms;android-34" "build-tools;34.0.0"
```

Tell the build where the SDK is by creating `android/local.properties`:

```properties
sdk.dir=/home/you/Android/Sdk
```

## Build

```sh
cd android
JAVA_HOME=~/jdk-17 ./gradlew assembleDebug
```

If you did not create `local.properties`, export `ANDROID_SDK_ROOT` instead
(`./gradlew` reads it too). The APK is written to
`app/build/outputs/apk/debug/app-debug.apk`. For a release build, run
`./gradlew assembleRelease` (signed with the debug key by default — configure a
signing config before publishing).

## Run locally

Start the agent as a web server from the repository root, listening so the
phone/emulator can reach it:

```sh
.venv/bin/python agent.py --serve --host 0.0.0.0 --port 8765
```

The client's default URL is `http://10.0.2.2:8765/chat`, which is the host
machine as seen from an Android **emulator**. On a **phone**, use the PC's LAN
address instead (e.g. `http://192.168.1.20:8765/chat`), with the phone and the
PC on the same network. Type the address into the field at the top of the app
and press **Save**.

You can check the server by hand:

```sh
curl -s http://127.0.0.1:8765/chat -H 'Content-Type: application/json' \
     -d '{"prompt": "hello"}'
```

### Configuration file

The default target is the bundled `app/src/main/assets/server.json`:

```json
{
  "url": "http://10.0.2.2:8765/chat",
  "timeoutSeconds": 60
}
```

`url` is the `/chat` endpoint, `timeoutSeconds` the connect/read timeout.
Editing the file changes the default for a fresh build; a URL saved in the app
takes precedence. **Clear** resets the conversation (forgets the session).

## Install on an Android device

`adb` lives in `$ANDROID_SDK_ROOT/platform-tools`; add it to your `PATH` once:

```sh
export PATH=$ANDROID_SDK_ROOT/platform-tools:$PATH
```

1. Enable **Developer options** (tap *Build number* seven times in
   *Settings > About phone*) and turn on **USB debugging**.
2. Connect the device over USB and check it is visible:

   ```sh
   adb devices
   ```

3. Install (or reinstall over an existing build):

   ```sh
   adb install -r app/build/outputs/apk/debug/app-debug.apk
   ```

   Or copy the APK to the device and open it, allowing installation from
   unknown sources when prompted.

4. Start the app (installing does not open it):

   ```sh
   adb shell am start -n com.example.httpposter/.MainActivity
   ```

5. In the app, set the server URL to the PC's LAN address, press **Save**, type
   a message and press **Send**. The reply appears below; errors (server down,
   timeout, bad URL) are shown in the response area.

### Run in an emulator (terminal)

The command-line tools can create and launch an emulator from the terminal.
Install the emulator and a system image, then create and start an AVD:

```sh
$ANDROID_SDK_ROOT/cmdline-tools/latest/bin/sdkmanager  "emulator" "system-images;android-34;google_apis;x86_64"

echo no | $ANDROID_SDK_ROOT/cmdline-tools/latest/bin/avdmanager create avd  -n pixel -k "system-images;android-34;google_apis;x86_64" -d pixel

$ANDROID_SDK_ROOT/emulator/emulator -avd pixel &
adb wait-for-device
adb install -r app/build/outputs/apk/debug/app-debug.apk

# adb install only puts the APK on the device; start the app explicitly
adb shell am start -n com.example.httpposter/.MainActivity
```

On the emulator the default URL `http://10.0.2.2:8765/chat` already reaches the
host machine, so it works without any change.

## Notes

- The device and the agent server must be on the same network; the agent must
  listen on `0.0.0.0` (not only `127.0.0.1`) for the phone to reach it.
- HTTP is used on purpose (the local agent has no TLS); the app enables
  cleartext traffic for it.
- `local.properties`, `.gradle/` and `build/` are not committed.
