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
st.caption("Best chops · adjust · hear the loop")

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
        end_bonus = 0.0
        if len(beat_times):
            end_dist = min(abs(end - bt) for bt in beat_times)
            end_bonus = max(0.0, 0.5 - end_dist * 2)
        balance = 1.0 - abs(energy - 0.15) * 2
        score = energy * (1 + bonus) * max(0.3, balance) * (1 + end_bonus)
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
                    "style": "Clean"
                })

    st.write(f"**{st.session_state.bpm:.1f} BPM**")

    out_dir = tempfile.mkdtemp()
    all_files = []

    for chop in st.session_state.auto_chops:
        st.markdown(f"### Chop {chop['id']}")

        start_adjust = st.slider("Start adjust (sec)", -1.0, 1.0, chop["start"] - chop["orig_start"], 0.05, key=f"start_{chop['id']}")
        chop["start"] = max(0, chop["orig_start"] + start_adjust)

        max_extend = min(4.0, duration - chop["orig_end"])
        max_shorten = max(0.3, chop["orig_end"] - chop["orig_start"] - 0.5)
        length_adjust = st.slider("Length adjust (sec)", -max_shorten, max_extend, chop["end"] - chop["orig_end"], 0.1, key=f"len_{chop['id']}")
        chop["end"] = chop["orig_end"] + length_adjust

        c1, c2, c3 = st.columns(3)
        with c1:
            chop["pitch"] = st.slider("Pitch", -12, 12, chop["pitch"], key=f"pitch_{chop['id']}")
        with c2:
            chop["reverse"] = st.checkbox("Reverse", value=chop["reverse"], key=f"rev_{chop['id']}")
        with c3:
            chop["style"] = st.selectbox("Style", ["Clean", "Dusty", "Warm", "Dark"],
                                         index=["Clean","Dusty","Warm","Dark"].index(chop["style"]),
                                         key=f"style_{chop['id']}")

        start_sample = int(max(0, chop["start"]) * sr)
        end_sample = int(min(chop["end"], duration) * sr)
        seg = y_full[start_sample:end_sample].copy()

        if chop["pitch"] != 0:
            seg =
