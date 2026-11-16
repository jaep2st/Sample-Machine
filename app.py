import streamlit as st
import os
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

st.set_page_config(page_title="Hip Hop Sample Machine", page_icon="🎧", layout="centered")

st.title("🎧 Hip Hop Sample Machine")
st.caption("Upload a song → get custom chops")

uploaded = st.file_uploader("Drop an audio file (MP3, WAV, M4A)", type=["mp3", "wav", "m4a", "flac"])

if uploaded:
    col1, col2 = st.columns(2)
    with col1:
        bars = st.slider("Bars per chop", 2, 16, 8)
    with col2:
        pitch = st.slider("Pitch (semitones)", -12, 12, -3)

    with st.spinner("Analyzing and chopping..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.getvalue())
            audio_path = tmp.name

        y, sr = librosa.load(audio_path, sr=44100, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        st.metric("BPM", f"{bpm:.1f}")

        bar_duration = (bars * 4 * 60) / bpm
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="time")
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

        candidates = []
        for start in onsets:
            end = start + bar_duration
            if end > len(y) / sr:
                continue
            mask = (times >= start) & (times <= end)
            if np.any(mask):
                energy = np.mean(rms[mask])
                candidates.append((start, end, energy))

        candidates.sort(key=lambda x: x[2], reverse=True)
        selected = []
        for start, end, energy in candidates:
            if all(not (start < s_end and end > s_start) for s_start, s_end, _ in selected):
                selected.append((start, end, energy))
            if len(selected) >= 8:
                break

        out_dir = tempfile.mkdtemp()
        samples = []

        for idx, (start, end, energy) in enumerate(selected, 1):
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            segment = y[start_sample:end_sample]
            base = f"chop_{idx:02d}_{bars}bars"

            # Original
            p = os.path.join(out_dir, f"{base}_original.wav")
            sf.write(p, segment, sr)
            samples.append(p)

            # Pitched
            if pitch != 0:
                pitched = librosa.effects.pitch_shift(segment, sr=sr, n_steps=pitch)
                p = os.path.join(out_dir, f"{base}_pitched_{pitch:+d}.wav")
                sf.write(p, pitched, sr)
                samples.append(p)

            # Filtered
            window = 15
            filtered = np.convolve(segment, np.ones(window)/window, mode="same")
            p = os.path.join(out_dir, f"{base}_filtered.wav")
            sf.write(p, filtered, sr)
            samples.append(p)

            # Reversed
            rev = segment[::-1]
            p = os.path.join(out_dir, f"{base}_reversed.wav")
            sf.write(p, rev, sr)
            samples.append(p)

    st.success(f"Found {len(selected)} strong {bars}-bar chops")

    for s in samples:
        name =
