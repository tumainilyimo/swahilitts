import io
import whisper
import torch
import ffmpeg
import torchaudio
import streamlit as st
LANGUAGES = {
    "en":"english",
    "zh":"chinese",
    "de":"german",
    "es":"spanish",
    "ru":"russian",
    "ko":"korean",
    "fr":"french",
    "ja":"japanese",
    "pt":"portuguese",
    "tr":"turkish",
    "pl":"polish",
    "ca":"catalan",
    "nl":"dutch",
    "ar":"arabic",
    "sv":"swedish",
    "it":"italian",
    "id":"indonesian",
    "hi":"hindi",
    "fi":"finnish",
    "vi":"vietnamese",
    "iw":"hebrew",
    "uk":"ukrainian",
    "el":"greek",
    "ms":"malay",
    "cs":"czech",
    "ro":"romanian",
    "da":"danish",
    "hu":"hungarian",
    "ta":"tamil",
    "no":"norwegian",
    "th":"thai",
    "ur":"urdu",
    "hr":"croatian",
    "bg":"bulgarian",
    "lt":"lithuanian",
    "la":"latin",
    "mi":"maori",
    "ml":"malayalam",
    "cy":"welsh",
    "sk":"slovak",
    "te":"telugu",
    "fa":"persian",
    "lv":"latvian",
    "bn":"bengali",
    "sr":"serbian",
    "az":"azerbaijani",
    "sl":"slovenian",
    "kn":"kannada",
    "et":"estonian",
    "mk":"macedonian",
    "br":"breton",
    "eu":"basque",
    "is":"icelandic",
    "hy":"armenian",
    "ne":"nepali",
    "mn":"mongolian",
    "bs":"bosnian",
    "kk":"kazakh",
    "sq":"albanian",
    "sw":"swahili",
    "gl":"galician",
    "mr":"marathi",
    "pa":"punjabi",
    "si":"sinhala",
    "km":"khmer",
    "sn":"shona",
    "yo":"yoruba",
    "so":"somali",
    "af":"afrikaans",
    "oc":"occitan",
    "ka":"georgian",
    "be":"belarusian",
    "tg":"tajik",
    "sd":"sindhi",
    "gu":"gujarati",
    "am":"amharic",
    "yi":"yiddish",
    "lo":"lao",
    "uz":"uzbek",
    "fo":"faroese",
    "ht":"haitian creole",
    "ps":"pashto",
    "tk":"turkmen",
    "nn":"nynorsk",
    "mt":"maltese",
    "sa":"sanskrit",
    "lb":"luxembourgish",
    "my":"myanmar",
    "bo":"tibetan",
    "tl":"tagalog",
    "mg":"malagasy",
    "as":"assamese",
    "tt":"tatar",
    "haw":"hawaiian",
    "ln":"lingala",
    "ha":"hausa",
    "ba":"bashkir",
    "jw":"javanese",
    "su":"sundanese",
}

def decode(model, mel, options):
    result = whisper.decode(model, mel, options)
    return result.text

def load_audio(audio):
    print(audio.type)
    if audio.type == "audio/wav" or audio.type == "audio/flac":
        wave, sr = torchaudio.load(audio)
        if sr != 16000:
            wave = torchaudio.transforms.Resample(sr, 16000)(wave)
        return wave.squeeze(0)

    elif audio.type == "audio/mpeg":
        audio = audio.read()
        audio, _ = (ffmpeg
            .input('pipe:0')
            .output('pipe:1', format='wav', acodec='pcm_s16le', ac=1, ar='16k')
            .run(capture_stdout=True, input=audio)
        )
        audio = io.BytesIO(audio)
        wave, sr = torchaudio.load(audio)
        if sr != 16000:
            wave = torchaudio.transforms.Resample(sr, 16000)(wave)
        return wave.squeeze(0)

    else:
        st.error("Unsupported audio format")

def detect_language(model, mel):
    _, probs = model.detect_language(mel)
    return max(probs, key=probs.get)

def main():

    st.title("Whisper ASR Demo")
    st.markdown(
            """
        This is a demo of OpenAI's Whisper ASR model. The model is trained on 680,000 hours of dataset. 
        """
    )

    model_selection = st.sidebar.selectbox("Select model", ["tiny", "base", "small", "medium", "large"])
    en_model_selection = st.sidebar.checkbox("English only model", value=False)

    if en_model_selection:
        model_selection += ".en"
    st.sidebar.write(f"Model: {model_selection+' (Multilingual)' if not en_model_selection else model_selection + ' (English only)'}")

    if st.sidebar.checkbox("Show supported languages", value=False):
            st.sidebar.info(list(LANGUAGES.values()))
    st.sidebar.title("Options")
    
    beam_size = st.sidebar.slider("Beam Size", min_value=1, max_value=10, value=5)
    fp16 = st.sidebar.checkbox("Enable FP16 for faster transcription (It may affect performance)", value=False)

    if not en_model_selection:
        task = st.sidebar.selectbox("Select task", ["transcribe", "translate (To English)"], index=0)
    else:
        task = st.sidebar.selectbox("Select task", ["transcribe"], index=0)

    st.title("Audio")
    audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "flac"])

    if audio_file is not None:
        st.audio(audio_file, format=audio_file.type)
        with st.spinner("Loading model..."):
            model = whisper.load_model(model_selection)
            model = model.to("cpu") if not torch.cuda.is_available() else model.to("cuda")
            

        audio = load_audio(audio_file)
        with st.spinner("Extracting features..."):
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
        if not en_model_selection:
            with st.spinner("Detecting language..."):
                language = detect_language(model, mel)
                st.markdown(f"Detected Language: {LANGUAGES[language]} ({language})")
        else:
            language = "en"
        configuration = {"beam_size": beam_size, "fp16": fp16, "task": task, "language": language}
        with st.spinner("Transcribing..."):
            options = whisper.DecodingOptions(**configuration)
            text = decode(model, mel, options)
        st.markdown(f"**Recognized Text:** {text}")

if __name__ == "__main__":
    main()
