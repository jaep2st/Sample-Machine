import streamlit as st
import os
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

st.set_page_config(page_title="Hip Hop Sample Machine", page_icon="🎧", layout="centered")
st.title("Hip Hop Sample Machine")
st.caption("Upload a song to get custom chops")

uploaded = st.file_uploader("Drop an audio file", type=["mp3", "wav", "m4a", "flac"])

if uploaded:
    bars = st.slider("Bars", 2, 16, 8)
    pitch = st.slider("Pitch", -12, 12, 0)
    preset = st.selectbox("Preset", ["Clean", "Dusty", "Dark"])

    with st.spinner("Analyzing..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.getvalue())
            audio_path = tmp.name

        y, sr = librosa.load(audio_path, sr=44100, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_idx = int(np.argmax(np.mean(chroma, axis=1)))
        keys = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        detected_key = keys[key_idx]
        st.write(f"BPM: {bpm:.1f} | Key: {detected_key}")

        bar_duration = (bars * 4 * 60) / bpm
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.times_like(rms, sr=sr)

        candidates = []
        for start in onsets:
            end = start + bar_duration
            if end > len(y)/sr:
                continue
            mask = (times >= start) & (times <= end)
            if np.any(mask):
                energy = float(np.mean(rms[mask]))
                candidates.append((start, end, energy))

        candidates.sort(key=lambda x: x[2], reverse=True)
        selected = []
        for start, end, energy in candidates:
            overlap = False
            for s_start, s_end, _ in selected:
                if start < s_end and end > s_start:
                    overlap = True
                    break
            if not overlap:
                selected.append((start, end, energy))
            if len(selected) >= 6:
                break

        out_dir = tempfile.mkdtemp()
        samples = []

        for idx, (start, end, energy) in enumerate(selected, 1):
            start_s = int(start * sr)
            end_s = int(end * sr)
            segment = y[start_s:end_s].copy()
            base = f"{int(bpm)}bpm_{detected_key}_chop{idx:02d}_{bars}bars"

            if preset == "Dusty":
                segment = np.convolve(segment, np.ones(25)/25, mode="same")
                segment = np.tanh(segment * 1.4) * 0.85
            elif preset == "Dark":
                segment = np.convolve(segment, np.ones(40)/40, mode="same")
                segment = segment * 0.7

            path = os.path.join(out_dir, base + "_original.wav")
            sf.write(path, segment, sr)
            samples.append(path)

            if pitch != 0:
                pitched = librosa.effects.pitch_shift(segment, sr=sr, n_steps=pitch)
                path = os.path.join(out_dir, base + f"_pitch{pitch}.wav")
                sf.write(path, pitched, sr)
                samples.append(path)

            rev = segment[::-1]
            path = os.path.join(out_dir, base + "_reversed.wav")
            sf.write(path, rev, sr)
            samples.append(path)

    st.success(f"Created {len(selected)} chops")

    for s in samples:
        st.audio(s)
        with open(s, "rb") as f:
            st.download_button("Download", f, file_name=Path(s).name, mime="audio/wav", key=s)

    zip_path = os.path.join(out_dir, "pack.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for s in samples:
            zf.write(s, Path(s).name)
    with open(zip_path, "rb") as f:
        st.download_button("Download Full Pack", f, file_name="sample_pack.zip")
