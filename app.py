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
st.caption("Best chops from the whole track — adjust length as needed")

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

def apply_fade(seg, fade_ms, sr):
    if fade_ms <= 0 or len(seg) == 0:
        return seg
    fade_samples = min(int(fade_ms / 1000 * sr), len(seg)//4)
    if fade_samples > 0:
        seg = seg.copy()
        seg[:fade_samples] *= np.linspace(0, 1, fade_samples)
        seg[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return seg

def norm(seg):
    peak = np.max(np.abs(seg))
    return seg / peak * 0.95 if peak > 0 else seg

def find_best_chops(y, sr, bars=4, max_chops=6):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)
    duration = (bars * 4 * 60) / bpm
    
    candidates = []
    for start in onsets:
        end = start + duration
        if end > len(y)/sr - 0.1:
            continue
        mask = (times >= start) & (times <= end)
        if not np.any(mask):
            continue
        energy = float(np.mean(rms[mask]))
        bonus = 0.0
        if len(beat_times):
            dist = min(abs(start - bt) for bt in beat_times)
            bonus = max(0.0, 1.0 - dist * 4)
        balance = 1.0 - abs(energy - 0.15) * 2
        score = energy * (1 + bonus) * max(0.3, balance)
        candidates.append((start, end, score))
    
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    selected = []
    for s, e, sc in candidates:
        if all(not (s < se and e > ss) for ss, se, _ in selected):
            selected.append((s, e, sc))
        if len(selected) >= max_chops:
            break
    
    return selected, bpm

uploaded = st.file_uploader("Drop a track", type=["mp3", "wav", "m4a", "flac"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.getvalue())
        path = tmp.name

    y_full, sr = librosa.load(path, sr=44100, mono=True)
    duration = len(y_full) / sr
    st.write(f"Track length: {duration:.1f}s")

    if "auto_chops" not in st.session_state or st.session_state.get("last_file") != uploaded.name:
        with st.spinner("Finding the best chops..."):
            selected, bpm = find_best_chops(y_full, sr, bars=4, max_chops=6)
            st.session_state.auto_chops = []
            st.session_state.bpm = bpm
            st.session_state.last_file = uploaded.name
            
            for i, (start, end, _) in enumerate(selected, 1):
                st.session_state.auto_chops.append({
                    "id": i,
                    "orig_start": start,
                    "orig_end": end,
                    "start": start,
                    "end": end,
                    "pitch": 0,
                    "reverse": False,
                    "style": "Clean",
                    "locked": False
                })

    st.write(f"**{st.session_state.bpm:.1f} BPM** · Adjust length on any chop")

    out_dir = tempfile.mkdtemp()
    all_files = []

    for chop in st.session_state.auto_chops:
        st.markdown(f"**Chop {chop['id']}**")

        # Length control — extend or shorten from the end
        max_extend = min(4.0, duration - chop["orig_end"])
        max_shorten = max(0.3, chop["orig_end"] - chop["orig_start"] - 0.5)
        length_adjust = st.slider(
            "Length adjust (sec)",
            -max_shorten, max_extend,
            chop["end"] - chop["orig_end"],
            0.1,
            key=f"len_{chop['id']}",
            help="Positive = extend, negative = shorten"
        )
        chop["end"] = chop["orig_end"] + length_adjust
        chop["start"] = chop["orig_start"]  # keep start fixed for now

        c1, c2, c3 = st.columns(3)
        with c1:
            chop["pitch"] = st.slider("Pitch", -12, 12, chop["pitch"], key=f"pitch_{chop['id']}")
        with c2:
            chop["reverse"] = st.checkbox("Reverse", value=chop["reverse"], key=f"rev_{chop['id']}")
        with c3:
            chop["style"] = st.selectbox("Style", ["Clean", "Dusty", "Warm", "Dark"],
                                         index=["Clean","Dusty","Warm","Dark"].index(chop["style"]),
                                         key=f"style_{chop['id']}")

        # Extract and process
        start_sample = int(chop["start"] * sr)
        end_sample = int(chop["end"] * sr)
        end_sample = min(end_sample, len(y_full))
        seg = y_full[start_sample:end_sample].copy()

        if chop["pitch"] != 0:
            seg = librosa.effects.pitch_shift(seg, sr=sr, n_steps=chop["pitch"])
        if chop["reverse"]:
            seg = seg[::-1]
        seg = process(seg, chop["style"])
        seg = apply_fade(seg, 12, sr)
        seg = norm(seg)

        name = f"chop_{chop['id']:02d}_{chop['start']:.1f}-{chop['end']:.1f}s"
        if chop["pitch"] != 0:
            name += f"_p{chop['pitch']:+d}"
        if chop["reverse"]:
            name += "_rev"
        if chop["style"] != "Clean":
            name += f"_{chop['style'].lower()}"
        name += ".wav"

        tmp_path = os.path.join(out_dir, name)
        sf.write(tmp_path, seg, sr)
        all_files.append(tmp_path)
        st.audio(tmp_path)
        with open(tmp_path, "rb") as f:
            st.download_button("Download", f, file_name=name, mime="audio/wav", key=f"dl_{chop['id']}")

    if all_files:
        zpath = os.path.join(out_dir, "best_chops.zip")
        with zipfile.ZipFile(zpath, "w") as zf:
            for fpath in all_files:
                zf.write(fpath, Path(fpath).name)
        with open(zpath, "rb") as f:
            st.download_button("Download All", f, file_name="best_chops.zip", key="pack")
