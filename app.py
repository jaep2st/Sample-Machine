import streamlit as st
import os
import tempfile
import zipfile
import json
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

st.set_page_config(page_title="Sample Machine", page_icon="🎧", layout="centered")
st.title("Sample Machine")
st.caption("Whole song · bar length · chops + one-shots")

def process(seg, style):
    if style == "Dusty":
        seg = np.convolve(seg, np.ones(15)/15, mode="same")
        seg = np.tanh(seg * 1.3) * 0.85
    elif style == "Warm":
        seg = np.convolve(seg, np.ones(5)/5, mode="same")
        seg = np.tanh(seg * 1.08) * 0.95
    elif style == "Dark":
        seg = np.convolve(seg, np.ones(25)/25, mode="same")
        seg = seg * 0.65
    return seg

def apply_eq(seg, sr, eq_type, amount):
    if eq_type == "None" or amount <= 0:
        return seg
    amount = max(0.1, amount)
    if eq_type == "Low shelf":
        low = np.convolve(seg, np.ones(21)/21, mode="same")
        return seg * (1 - amount * 0.4) + low * (1 + amount * 1.2)
    if eq_type == "High shelf":
        high = seg - np.convolve(seg, np.ones(5)/5, mode="same")
        return seg + high * amount * 2.5
    if eq_type == "Low-pass":
        kernel_size = int(7 + amount * 25)
        kernel = np.ones(kernel_size) / kernel_size
        return np.convolve(seg, kernel, mode="same")
    if eq_type == "High-pass":
        kernel_size = int(5 + amount * 15)
        low = np.convolve(seg, np.ones(kernel_size)/kernel_size, mode="same")
        return (seg - low) * (1 + amount * 0.8)
    return seg

def apply_fade(seg, fade_in_ms, fade_out_ms, sr):
    seg = seg.copy()
    if fade_in_ms > 0:
        fi = min(int(fade_in_ms / 1000 * sr), len(seg)//3)
        if fi > 0:
            seg[:fi] *= np.linspace(0, 1, fi)
    if fade_out_ms > 0:
        fo = min(int(fade_out_ms / 1000 * sr), len(seg)//3)
        if fo > 0:
            seg[-fo:] *= np.linspace(1, 0, fo)
    return seg

def apply_crossfade_loop(seg, sr, crossfade_ms=12):
    if crossfade_ms <= 0 or len(seg) < sr * 0.15:
        return np.concatenate([seg] * 4)
    cf = int(crossfade_ms / 1000 * sr)
    cf = min(cf, len(seg)//5)
    if cf <= 0:
        return np.concatenate([seg] * 4)
    fade_out = np.linspace(1, 0, cf)
    fade_in = np.linspace(0, 1, cf)
    looped = []
    for i in range(4):
        chunk = seg.copy()
        if i > 0:
            chunk[:cf] = chunk[:cf] * fade_in + seg[-cf:] * fade_out
        looped.append(chunk)
    return np.concatenate(looped)

def do_reverse(seg, sr):
    return np.flip(seg).copy()

def norm(seg, gain=1.0):
    peak = np.max(np.abs(seg))
    if peak > 0:
        seg = seg / peak * 0.92 * gain
    return np.clip(seg, -1.0, 1.0)

def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_idx = int(np.argmax(np.mean(chroma, axis=1)))
    return ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][key_idx]

def chop_whole_song(y, sr, bars=2):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    song_len = len(y) / sr
    bar_len = (bars * 4 * 60) / bpm
    chops = []
    start = 0.0
    i = 1
    while start + bar_len < song_len - 0.1:
        end = start + bar_len
        chops.append((start, end, i))
        start = end
        i += 1
    return chops, bpm

def extract_oneshots(y, sr, max_shots=12):
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True, delta=0.05)
    shots = []
    for onset in onsets[:200]:
        window = y[int(onset*sr):int((onset+0.18)*sr)]
        if len(window) < 40:
            continue
        hit = window[:int(0.08*sr)]
        tail = window[int(0.08*sr):]
        hit_energy = float(np.mean(np.abs(hit))) if len(hit) else 0
        tail_energy = float(np.mean(np.abs(tail))) if len(tail) else 0
        peak = float(np.max(np.abs(window)))
        if hit_energy > 0.08 and peak > 0.25 and hit_energy > tail_energy * 1.3:
            shots.append((onset, hit_energy * peak))
    shots.sort(key=lambda x: x[1], reverse=True)
    return shots[:max_shots]

uploaded = st.file_uploader("Drop a track", type=["mp3", "wav", "m4a", "flac"])

if uploaded:
    suffix = Path(uploaded.name).suffix.lower()
    if suffix not in [".mp3", ".wav", ".m4a", ".flac"]:
        suffix = ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        path = tmp.name

    try:
        y_full, sr = librosa.load(path, sr=44100, mono=True)
    except Exception as e:
        st.error(f"Could not load audio: {e}")
        st.stop()

    duration = len(y_full) / sr
    key = detect_key(y_full, sr)
    st.write(f"Track length: {duration:.1f}s · **{key}**")

    bar_choice = st.radio("Bar length", ["2 bars", "4 bars", "8 bars", "16 bars"], index=0, horizontal=True)
    bars = int(bar_choice.split()[0])

    tab_chops, tab_oneshots = st.tabs(["Chops", "One-shots"])

    with tab_chops:
        with st.spinner("Chopping the whole song..."):
            chops, bpm = chop_whole_song(y_full, sr, bars=bars)
        st.write(f"**{bpm:.1f} BPM** · {len(chops)} chops")

        out_dir = tempfile.mkdtemp()
        included = []

        for start, end, i in chops:
            seg = y_full[int(start*sr):int(end*sr)].copy()
            name = f"{int(bpm)}bpm_{key}_chop{i:02d}_{start:.1f}s.wav"
            tmp_path = os.path.join(out_dir, name)
            sf.write(tmp_path, seg, sr)
            included.append(tmp_path)

            st.markdown(f"**Chop {i}** · from {start:.1f}s")
            st.audio(tmp_path)
            with open(tmp_path, "rb") as f:
                st.download_button("Download", f, file_name=name, mime="audio/wav", key=f"dl_{i}")

            looped = apply_crossfade_loop(seg, sr)
            loop_path = os.path.join(out_dir, name.replace(".wav", "_loop.wav"))
            sf.write(loop_path, looped, sr)
            st.audio(loop_path)

            with st.expander("Tweak", expanded=False):
                pitch = st.slider("Pitch", -12, 12, 0, key=f"p_{i}")
                rev = st.checkbox("Reverse", key=f"r_{i}")
                style = st.selectbox("Style", ["Clean", "Dusty", "Warm", "Dark"], key=f"s_{i}")
                if st.button("Apply", key=f"apply_{i}"):
                    new_seg = seg.copy()
                    if pitch != 0:
                        new_seg = librosa.effects.pitch_shift(new_seg, sr=sr, n_steps=pitch)
                    if rev:
                        new_seg = do_reverse(new_seg, sr)
                    new_seg = process(new_seg, style)
                    new_seg = norm(new_seg)
                    new_path = os.path.join(out_dir, f"tweaked_{i}.wav")
                    sf.write(new_path, new_seg, sr)
                    st.audio(new_path)

        if included:
            zpath = os.path.join(out_dir, "all_chops.zip")
            with zipfile.ZipFile(zpath, "w") as zf:
                for fpath in included:
                    zf.write(fpath, Path(fpath).name)
            with open(zpath, "rb") as f:
                st.download_button("Download All Chops", f, file_name="all_chops.zip", key="allchops")

    with tab_oneshots:
        oneshots = extract_oneshots(y_full, sr)
        st.write(f"{len(oneshots)} one-shots")
        os_files = []
        for i, (onset, energy) in enumerate(oneshots, 1):
            window = y_full[int(onset*sr):int((onset+0.5)*sr)].copy()
            window = norm(window)
            os_name = f"{int(bpm)}bpm_{key}_oneshot{i:02d}_{onset:.1f}s.wav"
            os_path = os.path.join(out_dir, os_name)
            sf.write(os_path, window, sr)
            os_files.append(os_path)
            st.markdown(f"**One-shot {i}** · from {onset:.1f}s")
            st.audio(os_path)
            with open(os_path, "rb") as f:
                st.download_button("Download", f, file_name=os_name, mime="audio/wav", key=f"osdl_{i}")
            loop_os = np.concatenate([window] * 4)
            loop_os_path = os.path.join(out_dir, f"osloop_{i}.wav")
            sf.write(loop_os_path, loop_os, sr)
            st.audio(loop_os_path)

        if os_files:
            zpath = os.path.join(out_dir, "oneshots.zip")
            with zipfile.ZipFile(zpath, "w") as zf:
                for fpath in os_files:
                    zf.write(fpath, Path(fpath).name)
            with open(zpath, "rb") as f:
                st.download_button("Download All One-shots", f, file_name="oneshots.zip", key="allos")
