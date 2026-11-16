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
st.caption("For producers who actually flip samples")

uploaded = st.file_uploader("Drop a track", type=["mp3", "wav", "m4a", "flac"])

if uploaded:
    c1, c2, c3 = st.columns(3)
    with c1:
        bars = st.slider("Bars", 1, 8, 4)
    with c2:
        pitch = st.slider("Pitch", -12, 12, 0)
    with c3:
        style = st.selectbox("Style", ["Clean", "Dusty", "Warm", "Dark"])

    max_chops = st.slider("Chops", 3, 8, 4)
    normalize = st.checkbox("Normalize for sampler", value=True)

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
                seg = np.convolve(seg, np.ones(31)/31, mode="same")
                seg *= 0.7
            return seg

        def norm(seg):
            if not normalize:
                return seg
            peak = np.max(np.abs(seg))
            if peak > 0:
                return seg / peak * 0.95
            return seg

        for i, (start, end, _) in enumerate(selected, 1):
            seg = y[int(start*sr):int(end*sr)].copy()
            seg = process(seg, style)
            seg = norm(seg)
            base = f"{int(bpm)}bpm_{key}_{style.lower()}_{i:02d}"

            # Loop
            p = os.path.join(out, base + "_loop.wav")
            sf.write(p, seg, sr)
            files.append(p)

            # One-shot from first transient
            env = np.abs(seg)
            peak_idx = np.argmax(env[:int(sr*0.4)])
            oneshot = seg[max(0, peak_idx-int(sr*0.015)):peak_idx+int(sr*1.5)]
            oneshot = norm(oneshot)
            p = os.path.join(out, base + "_oneshot.wav")
            sf.write(p, oneshot, sr)
            files.append(p)

            if pitch != 0:
                pitched = librosa.effects.pitch_shift(seg, sr=sr, n_steps=pitch)
                pitched = norm(pitched)
                p = os.path.join(out, base + f"_p{pitch:+d}.wav")
                sf.write(p, pitched, sr)
                files.append(p)

    st.success(f"{len(selected)} chops")

    for fpath in files:
        name = Path(fpath).name
        st.caption(name)
        st.audio(fpath)
        with open(fpath, "rb") as fh:
            st.download_button("Save", fh, file_name=name, mime="audio/wav", key=name)

    zpath = os.path.join(out, "pack.zip")
    with zipfile.ZipFile(zpath, "w") as zf:
        for fpath in files:
            zf.write(fpath, Path(fpath).name)
    with open(zpath, "rb") as fh:
        st.download_button("Download Full Pack", fh, file_name="sample_pack.zip")
