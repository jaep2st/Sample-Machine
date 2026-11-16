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
st.caption("Phrase-aware chops")

uploaded = st.file_uploader("Drop a track", type=["mp3", "wav", "m4a", "flac"])

if uploaded:
    c1, c2, c3 = st.columns(3)
    with c1:
        bars = st.slider("Bars", 1, 16, 4)
    with c2:
        pitch = st.slider("Pitch", -12, 12, 0)
    with c3:
        mode = st.selectbox("Mode", ["Loop", "One-shot", "Both"])

    style = st.selectbox("Style", ["Clean", "Dusty", "Warm", "Dark"])
    max_chops = st.slider("How many", 3, 10, 5)

    with st.spinner("Finding phrases..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded.getvalue())
            path = tmp.name

        y, sr = librosa.load(path, sr=44100, mono=True)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        beat_times = librosa.frames_to_time(beats, sr=sr)

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][int(np.argmax(np.mean(chroma, axis=1)))]
        st.write(f"**{bpm:.1f} BPM** · **{key}**")

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
            # Prefer starts near beats
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
                seg = np.convolve(seg, np.ones(19)/19, mode="same")
                seg = np.tanh(seg * 1.3) * 0.87
            elif style == "Warm":
                seg = np.convolve(seg, np.ones(9)/9, mode="same")
                seg = np.tanh(seg * 1.12) * 0.93
            elif style == "Dark":
                seg = np.convolve(seg, np.ones(35)/35, mode="same")
                seg *= 0.68
            return seg

        for i, (start, end, _) in enumerate(selected, 1):
            seg = y[int(start*sr):int(end*sr)].copy()
            seg = process(seg, style)
            base = f"{int(bpm)}bpm_{key}_{style.lower()}_{mode.lower()}_{i:02d}"

            if mode in ("Loop", "Both"):
                p = os.path.join(out, base + "_loop.wav")
                sf.write(p, seg, sr)
                files.append(p)
            if mode in ("One-shot", "Both"):
                # Shorter one-shot from the start of the phrase
                oneshot = seg[:int(sr * min(2.5, len(seg)/sr))]
                p = os.path.join(out, base + "_oneshot.wav")
                sf.write(p, oneshot, sr)
                files.append(p)
            if pitch != 0 and mode != "One-shot":
                pitched = librosa.effects.pitch_shift(seg, sr=sr, n_steps=pitch)
                p = os.path.join(out, base + f"_p{pitch:+d}.wav")
                sf.write(p, pitched, sr)
                files.append(p)

    st.success(f"{len(selected)} chops")

    for fpath in files:
        name = Path(fpath).name
        st.audio(fpath)
        with open(fpath, "rb") as fh:
            st.download_button(name, fh, file_name=name, mime="audio/wav", key=name)

    zpath = os.path.join(out, "pack.zip")
    with zipfile.ZipFile(zpath, "w") as zf:
        for fpath in files:
            zf.write(fpath, Path(fpath).name)
    with open(zpath, "rb") as fh:
        st.download_button("Download Pack", fh, file_name="sample_pack.zip")
