import streamlit as st
import os
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

st.set_page_config(page_title="Sample Machine", page_icon="🎧", layout="centered")

st.title("Sample Machine")
st.caption("Musical chops for producers")

uploaded = st.file_uploader("Drop a track", type=["mp3", "wav", "m4a", "flac"])

if uploaded:
    c1, c2, c3 = st.columns(3)
    with c1:
        bars = st.slider("Bars", 2, 16, 8)
    with c2:
        pitch = st.slider("Pitch", -12, 12, 0)
    with c3:
        style = st.selectbox("Style", ["Clean", "Dusty", "Dark", "Warm", "Tape"])

    max_chops = st.slider("Max chops", 4, 12, 6)

    with st.spinner("Finding strong phrases..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.getvalue())
            audio_path = tmp.name

        y, sr = librosa.load(audio_path, sr=44100, mono=True)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key_idx = int(np.argmax(np.mean(chroma, axis=1)))
        keys = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        detected_key = keys[key_idx]

        st.write(f"**{bpm:.1f} BPM**  ·  **{detected_key}**")

        # Prefer onsets near beats
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        rms = librosa.feature.rms(y=y)[0]
        times = librosa.times_like(rms, sr=sr)

        bar_duration = (bars * 4 * 60) / bpm
        candidates = []

        for start in onsets:
            # Score higher if close to a beat
            beat_dist = min(abs(start - bt) for bt in beat_times) if len(beat_times) else 1
            beat_bonus = max(0, 1 - beat_dist * 4)

            end = start + bar_duration
            if end > len(y) / sr - 0.1:
                continue
            mask = (times >= start) & (times <= end)
            if not np.any(mask):
                continue
            energy = float(np.mean(rms[mask]))
            score = energy * (1 + beat_bonus * 0.6)
            candidates.append((start, end, score))

        candidates.sort(key=lambda x: x[2], reverse=True)
        selected = []
        for start, end, score in candidates:
            if all(not (start < s_end and end > s_start) for s_start, s_end, _ in selected):
                selected.append((start, end, score))
            if len(selected) >= max_chops:
                break

        out_dir = tempfile.mkdtemp()
        samples = []

        def apply_style(seg, style):
            if style == "Dusty":
                seg = np.convolve(seg, np.ones(21)/21, mode="same")
                seg = np.tanh(seg * 1.35) * 0.88
            elif style == "Dark":
                seg = np.convolve(seg, np.ones(45)/45, mode="same")
                seg *= 0.65
            elif style == "Warm":
                seg = np.convolve(seg, np.ones(11)/11, mode="same")
                seg = np.tanh(seg * 1.15) * 0.92
            elif style == "Tape":
                seg = np.convolve(seg, np.ones(9)/9, mode="same")
                noise = np.random.normal(0, 0.008, len(seg))
                seg = seg * 0.95 + noise
                seg = np.tanh(seg * 1.2) * 0.9
            return seg

        for idx, (start, end, score) in enumerate(selected, 1):
            s0 = int(start * sr)
            s1 = int(end * sr)
            segment = y[s0:s1].copy()
            segment = apply_style(segment, style)

            base = f"{int(bpm)}bpm_{detected_key}_{style.lower()}_chop{idx:02d}_{bars}bars"

            path = os.path.join(out_dir, base + ".wav")
            sf.write(path, segment, sr)
            samples.append(path)

            if pitch != 0:
                pitched = librosa.effects.pitch_shift(segment, sr=sr, n_steps=pitch)
                path = os.path.join(out_dir, base + f"_p{pitch:+d}.wav")
                sf.write(path, pitched, sr)
                samples.append(path)

            # One reversed version per chop
            if idx <= 3:
                rev = segment[::-1]
                path = os.path.join(out_dir, base + "_rev.wav")
                sf.write(path, rev, sr)
                samples.append(path)

    st.success(f"{len(selected)} strong chops")

    for s in samples:
        name = Path(s).name
        st.audio(s)
        with open(s, "rb") as f:
            st.download_button(name, f, file_name=name, mime="audio/wav", key=name)

    zip_path = os.path.join(out_dir, "pack.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for s in samples:
            zf.write(s, Path(s).name)

    with open(zip_path, "rb") as f:
        st.download_button("Download Pack", f, file_name="sample_pack.zip", mime="application/zip")
