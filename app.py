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
st.caption("Best chops from the whole track, instantly")

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

def find_best_chops(y, sr, bars=4, max_chops=6, mode="Phrase"):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)
    
    if mode == "Phrase":
        duration = (bars * 4 * 60) / bpm
    else:
        duration = (bars * 60) / bpm
    
    candidates = []
    for start in onsets:
        end = start + duration
        if end > len(y)/sr - 0.1:
            continue
        mask = (times >= start) & (times <= end)
        if not np.any(mask):
            continue
        energy = float(np.mean(rms[mask]))
        # Prefer onsets close to a beat
        bonus = 0.0
        if len(beat_times):
            dist = min(abs(start - bt) for bt in beat_times)
            bonus = max(0.0, 1.0 - dist * 4)
        # Prefer mid-energy (not silent, not crushed)
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

    # Immediate best chops from full track
    if "auto_chops" not in st.session_state or st.session_state.get("last_file") != uploaded.name:
        with st.spinner("Finding the best chops..."):
            selected, bpm = find_best_chops(y_full, sr, bars=4, max_chops=6)
            st.session_state.auto_chops = []
            st.session_state.bpm = bpm
            st.session_state.last_file = uploaded.name
            
            for i, (start, end, _) in enumerate(selected, 1):
                seg = y_full[int(start*sr):int(end*sr)].copy()
                seg = apply_fade(seg, 12, sr)
                seg = norm(seg)
                st.session_state.auto_chops.append({
                    "id": i,
                    "start": start,
                    "end": end,
                    "audio": seg,
                    "pitch": 0,
                    "reverse": False,
                    "style": "Clean",
                    "locked": False
                })

    st.write(f"**{st.session_state.bpm:.1f} BPM** · Best chops from the full track")

    out_dir = tempfile.mkdtemp()
    all_files = []

    for chop in st.session_state.auto_chops:
        col1, col2 = st.columns([5, 1])
        with col1:
            st.markdown(f"**Chop {chop['id']}** · {chop['start']:.1f}s")
        with col2:
            chop["locked"] = st.checkbox("Lock", value=chop["locked"], key=f"lock_{chop['id']}")

        # Controls
        c1, c2, c3 = st.columns(3)
        with c1:
            chop["pitch"] = st.slider("Pitch", -12, 12, chop["pitch"], key=f"pitch_{chop['id']}")
        with c2:
            chop["reverse"] = st.checkbox("Reverse", value=chop["reverse"], key=f"rev_{chop['id']}")
        with c3:
            chop["style"] = st.selectbox("Style", ["Clean", "Dusty", "Warm", "Dark"], 
                                         index=["Clean","Dusty","Warm","Dark"].index(chop["style"]), 
                                         key=f"style_{chop['id']}")

        # Process
        seg = chop["audio"].copy()
        if chop["pitch"] != 0:
            seg = librosa.effects.pitch_shift(seg, sr=sr, n_steps=chop["pitch"])
        if chop["reverse"]:
            seg = seg[::-1]
        seg = process(seg, chop["style"])
        seg = norm(seg)

        # Preview + download
        name = f"chop_{chop['id']:02d}_{chop['start']:.1f}s"
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

    # Pack
    if all_files:
        zpath = os.path.join(out_dir, "best_chops.zip")
        with zipfile.ZipFile(zpath, "w") as zf:
            for fpath in all_files:
                zf.write(fpath, Path(fpath).name)
        with open(zpath, "rb") as f:
            st.download_button("Download All Best Chops", f, file_name="best_chops.zip", key="pack")

    st.divider()
    st.subheader("Want a different section?")
    region_start = st.slider("Start (sec)", 0.0, max(0.1, duration-1), 0.0, 0.1)
    region_end = st.slider("End (sec)", 1.0, duration, min(duration, 30.0), 0.1)
    
    if st.button("Chop this section instead"):
        y_region = y_full[int(region_start*sr):int(region_end*sr)]
        selected, bpm = find_best_chops(y_region, sr, bars=4, max_chops=4)
        st.success(f"Found {len(selected)} chops in this section")
        # (You can expand this later to replace or append)
    
