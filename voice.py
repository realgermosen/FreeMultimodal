from gtts import gTTS
import tempfile
import os
import io

def text_to_speech(text):
    tts = gTTS(text=text, lang='en', slow=False)

    # Save the generated speech to a temporary file
    audio_tempfile = tempfile.NamedTemporaryFile(delete=False)
    tts.save(audio_tempfile.name)
    audio_tempfile.close()

    # Read the file as bytes directly
    with open(audio_tempfile.name, "rb") as f:
        audio_bytes = f.read()

    # Delete the temporary file
    os.unlink(audio_tempfile.name)

    return audio_bytes
