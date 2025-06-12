import streamlit as st
import tempfile
import os
from faster_whisper import WhisperModel
import srt
from datetime import timedelta

# Carrega o modelo Whisper (medium) no dispositivo correto
# WhisperModel("base", download_root="./models")
model = WhisperModel("base", device='cpu', compute_type="int8", download_root="./models", local_files_only=True)

st.title("🎙️ Crie Legendas Automáticas para seus Áudios")
st.write("📤 Envie um arquivo de áudio (.mp3 ou .wav)")

uploaded_file = st.file_uploader("Arraste e solte um arquivo aqui", type=["mp3", "wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/mp3")

    if st.button("Transcrever"):
        with st.spinner("Transcrevendo..."):
            # Salva o arquivo temporariamente
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Executa a transcrição com segmentação
            segments, info = model.transcribe(tmp_path, beam_size=5, language="pt")

            transcribed_text = ""
            subtitles = []

            for i, segment in enumerate(segments):
                start = timedelta(seconds=segment.start)
                end = timedelta(seconds=segment.end)
                text = segment.text.strip()

                transcribed_text += " " + text
                subtitles.append(srt.Subtitle(index=i + 1, start=start, end=end, content=text))

            # Mostra o texto completo transcrito
            st.subheader("📝 Transcrição")
            st.write(transcribed_text.strip())

            # Gera string SRT
            srt_string = srt.compose(subtitles)

            # Salva para download
            srt_path = os.path.join(tempfile.gettempdir(), "transcricao.srt")
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(srt_string)

            with open(srt_path, "rb") as f:
                st.download_button("📥 Baixar legenda .srt", data=f, file_name="transcricao.srt", mime="text/plain")
