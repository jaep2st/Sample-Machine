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
st.caption("Target the parts you actually want")

uploaded = st.file_uploader("Drop a track", type=["mp3", "wav", "m4a", "flac"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.getvalue())
        audio_path = tmp.name

    y_full, sr = librosa.load(audio_path, sr=44100, mono=True)
    duration = len(y_full) / sr

    st.write(f"Track length: **{duration:.1f}s**")

    c1, c2 = st.columns(2)
    with c1:
        region_start = st.slider("Start (seconds)", 0.0, max(0.1, duration - 1), 0.0, 0.1)
    with c2:
        region_end = st.slider("End (seconds)", 0.1, duration, min(duration, 30.0), 0.1)

    if region_end <= region_start:
        st.warning("End must be after start")
        st.stop()

    c3, c4, c5 = st.columns(3)
    with c3:
        bars = st.slider("Bars", 1, 8, 4)
    with c4:
        pitch = st.slider("Pitch", -12, 12, 0)
    with c5:
        style = st.selectbox("Style", ["Clean", "Dusty", "Warm", "Dark"])

    max_chops = st.slider("Chops from this section", 1, 8, 3)
    normalize = st.checkbox("Normalize", value=True)

    if st.button("Chop this section", type="primary"):
        with st.spinner("Chopping selected region..."):
            start_sample = int(region_start * sr)
            end_sample = int(region_end * sr)
            y = y_full[start_sample:end_sample]

            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
            beat_times = librosa.frames_to_time(beats, sr=sr)

            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            key = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][int(np.argmax(np.mean(chroma, axis=1)))]
            st.write(f"**{bpm:.1f} BPM** · **{key}** · Section: {region_start:.1f}s–{region_end:.1f}s")

            onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
            rms = librosa.feature.rms(y=y)[0]
            times = librosa.times_like(rms, sr=sr)
            bar_dur = (bars * 4 * 60) / bpm

            cands = []
            for start in onsets:
                end = start + bar_dur
                if end > len(y)/sr - 0.05:
                    continue
                mask = (times >= start) & (times <= end)
                if not np.any(mask):
                    continue
                energy = float(np.mean(rms[mask]))
                if len(beat_times):
                    dist = min(abs(start - bt) for bt in beat_times)
                    bonus = max(0, 1 - dist * 5)
                else:
                    bonus = 0
                cands.append((start, end, energy * (1 + bonus)))

            cands.sort(key=lambda x: x[2], reverse=True)
            selected = []
            for s, e, sc in cands:
                if all(not (s < se and e > ss) for ss, se, _ in selected):
                    selected.append((s, e, sc))
                if len(selected) >= max_chops:
                    break

            out = tempfile.mkdtemp()
            files = []

            def process(seg, style):
                if style == "Dusty":
                    seg = np.convolve(seg, np.ones(17)/17, mode="same")
                    seg = np.tanh(seg * 1.25) * 0.88
                elif style == "Warm":
                    seg = np.convolve(seg, np.ones(7)/7, mode="same")
                    seg = np.tanh(seg * 1.1) * 0.94
                elif style == "Dark":
                    seg =
