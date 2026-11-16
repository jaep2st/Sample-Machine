import streamlit as st
import os
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sample Machine", page_icon="🎧", layout="centered")
st.title("Sample Machine")
st.caption("Target sections and refine chops")

uploaded = st.file_uploader("Drop a track", type=["mp3", "wav", "m4a", "flac"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.getvalue())
        audio_path = tmp.name

    y_full, sr = librosa.load(audio_path, sr=44100, mono=True)
    duration = len(y_full) / sr
    st.write(f"Length: {duration:.1f}s")

    region_start = st.slider("Start (sec)", 0.0, max(0.1, duration-1), 0.0, 0.1)
    region_end = st.slider("End (sec)", 0.1, duration, min(duration, 30.0), 0.1)

    if region_end <= region_start:
        st.warning("End must be after start")
        st.stop()

    y_region = y_full[int(region_start*sr):int(region_end*sr)]
    fig, ax = plt.subplots(figsize=(10, 2))
    times = np.linspace(region_start, region_end, len(y_region))
    ax.plot(times, y_region, color="#e74c3c", linewidth=0.5)
    ax.set_xlim(region_start, region_end)
    ax.set_ylim(-1, 1)
    ax.axis("off")
    st.pyplot(fig)
    plt.close()

    bars = st.slider("Bars", 1, 8, 4)
    pitch = st.slider("Pitch", -12, 12, 0)
    style = st.selectbox("Style", ["Clean", "Dusty", "Warm", "Dark"])
    max_chops = st.slider("Chops", 1, 8, 3)
    normalize = st.checkbox("Normalize", value=True)

    if st.button("Chop this section", type="primary"):
        with st.spinner("Chopping..."):
            y = y_region.copy()
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
            beat_times = librosa.frames_to_time(beats, sr=sr)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            key = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][int(np.argmax(np.mean(chroma, axis=1)))]

            onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
            rms = librosa.feature.rms(y=y)[0]
            t = librosa.times_like(rms, sr=sr)
            bar_dur = (bars * 4 * 60) / bpm

            cands = []
            for start in onsets:
                end = start + bar_dur
                if end > len(y)/sr - 0.05:
                    continue
                mask = (t >= start) & (t <= end)
                if not np.any(mask):
                    continue
                energy = float(np.mean(rms[mask]))
                bonus = 0.0
                if len(beat_times):
                    dist = min(abs(start - bt) for bt in beat_times)
                    bonus = max(0.0, 1.0 - dist * 5)
                cands.append((start, end, energy * (1 + bonus)))

            cands.sort(key=lambda x: x[2], reverse=True)
            selected = []
            for s, e, sc in cands:
                if all(not (s < se and e > ss) for ss, se, _ in selected):
                    selected.append((s, e, sc))
                if len(selected) >= max_chops:
                    break

            def process(seg, style):
                if style == "Dusty":
                    seg = np.convolve(seg, np.ones(17)/17, mode="same")
                    seg = np.tanh(seg * 1.25) * 0.88
                elif style == "Warm":
                    seg = np.convolve(seg, np.ones(7)/7, mode="same")
                    seg = np.tanh(seg * 1.1) * 0.94
                elif style == "Dark":
                    seg = np.convolve(seg, np.ones(31)/31, mode="same")
                    seg = seg * 0.7
                return seg

            def norm(seg):
                if not normalize or len(seg) == 0:
                    return seg
                peak = np.max(np.abs(seg))
                return seg / peak * 0.95 if peak > 0 else seg

            out = tempfile.mkdtemp()
            files = []

            st.write(f"**{bpm:.1f} BPM** · **{key}**")
            st.success(f"{len(selected)} chops")

            for i, (start, end, _) in enumerate(selected, 1):
                seg = y[int(start*sr):int(end*sr)].copy()
                seg = process(seg, style)
                seg = norm(seg)
                abs_start = region_start + start
                base = f"{int(bpm)}bpm_{key}_{style.lower()}_{i:02d}_{abs_start:.1f}s"

                loop_path = os.path.join(out, base + "_loop.wav")
                sf.write(loop_path, seg, sr)
