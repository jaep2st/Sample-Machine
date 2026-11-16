import streamlit as st
import os
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import subprocess

st.set_page_config(page_title="Sample Machine", page_icon="🎧", layout="centered")
st.title("Sample Machine")
st.caption("Best chops · stems · loop preview · one-shots")

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

def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_idx = int(np.argmax(np.mean(chroma, axis=1)))
    return ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][key_idx]

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

def extract_oneshots(y, sr, max_shots=4):
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    shots = []
    for onset in onsets[:40]:
        window = y[int(onset*sr):int((onset+0.35)*sr)]
        if len(window) < 100:
            continue
        energy = float(np.mean(np.abs(window)))
        if energy > 0.09:
            shots.append((onset, energy))
    shots.sort(key=lambda x: x[1], reverse=True)
    return shots[:max_shots]

def separate_stems(audio_path, out_dir):
    try:
        cmd = ["python", "-m", "demucs", "-n", "htdemucs", "-o", out_dir, audio_path]
        subprocess.run(cmd, check=True, capture_output=True)
        # Demucs outputs to out_dir/htdemucs/<filename>/
        base = Path(audio_path).stem
        stem_dir = Path(out_dir) / "htdemucs" / base
        stems = {}
        for stem in ["vocals", "drums", "bass", "other"]:
            p = stem_dir / f"{stem}.wav"
            if p.exists():
                stems[stem] = str(p)
        return stems
    except Exception as e:
        st.error(f"Stem separation failed: {e}")
        return {}

uploaded = st.file_uploader("Drop a track", type=["mp3", "wav", "m4a", "flac"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.getvalue())
        path = tmp.name

    y_full, sr = librosa.load(path, sr=44100, mono=True)
    duration = len(y_full) / sr
    key = detect_key(y_full, sr)
    st.write(f"Track length: {duration:.1f}s · **{key}**")

    use_stems = st.checkbox("Separate stems first (slower)", value=False)
    stem_choice = "full"
    stems = {}

    if use_stems:
        with st.spinner("Separating stems with Demucs..."):
            stem_out = tempfile.mkdtemp()
            stems = separate_stems(path, stem_out)
        if stems:
            stem_choice = st.selectbox("Chop from", ["full", "drums", "bass", "vocals", "other"])
        else:
            st.warning("Stem separation unavailable — using full mix")

    # Choose source audio
    if stem_choice != "full" and stem_choice in stems:
        y_source, sr = librosa.load(stems[stem_choice], sr=44100, mono=True)
        source_label = stem_choice
    else:
        y_source = y_full
        source_label = "full"

    if "auto_chops" not in st.session_state or st.session_state.get("last_file") != uploaded.name or st.session_state.get("stem") != stem_choice:
        with st.spinner("Finding the best chops..."):
            selected, bpm = find_best_chops(y_source, sr, bars=4, max_chops=6)
            st.session_state.auto_chops = []
            st.session_state.bpm = bpm
            st.session_state.key = key
            st.session_state.last_file = uploaded.name
            st.session_state.stem = stem_choice
            
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

    bpm = st.session_state.bpm
    st.write(f"**{bpm:.1f} BPM** · **{key}** · {source_label}")

    out_dir = tempfile.mkdtemp()
    all_files = []

    for chop in st.session_state.auto_chops:
        st.markdown(f"### Chop {chop['id']}")

        start_adjust = st.slider("Start adjust (sec)", -1.0, 1.0, chop["start"] - chop["orig_start"], 0.05, key=f"start_{chop['id']}")
        chop["start"] = max(0, chop["orig_start"] + start_adjust)

        max_extend = min(4.0, len(y_source)/sr - chop["orig_end"])
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
        end_sample = int(min(chop["end"], len(y_source)/sr) * sr)
        seg = y_source[start_sample:end_sample].copy()

        if chop["pitch"] != 0:
            seg = librosa.effects.pitch_shift(seg, sr=sr, n_steps=chop["pitch"])
        if chop["reverse"]:
            seg = seg[::-1]
        seg = process(seg, chop["style"])
        seg = apply_fade(seg, 12, sr)
        seg = norm(seg)

        name = f"{int(bpm)}bpm_{key}_{source_label}_chop{chop['id']:02d}"
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

        st.markdown("**Single**")
        st.audio(tmp_path)
        with open(tmp_path, "rb") as f:
            st.download_button("Download single", f, file_name=name, mime="audio/wav", key=f"dl_{chop['id']}")

        looped = np.concatenate([seg] * 4)
        loop_path = os.path.join(out_dir, f"loop_{chop['id']}.wav")
        sf.write(loop_path, looped, sr)
        st.markdown("**Loop (4x)**")
        st.audio(loop_path)

    # One-shots from the chosen source
    st.divider()
    st.subheader("One-shots")
    oneshots = extract_oneshots(y_source, sr, max_shots=4)
    for i, (onset, energy) in enumerate(oneshots, 1):
        window = y_source[int(onset*sr):int((onset+0.5)*sr)]
        window = apply_fade(window, 6, sr)
        window = norm(window)
        os_name = f"{int(bpm)}bpm_{key}_{source_label}_oneshot{i:02d}_{onset:.1f}s.wav"
        os_path = os.path.join(out_dir, os_name)
        sf.write(os_path, window, sr)
        all_files.append(os_path)
        st.markdown(f"**One-shot {i}** · {onset:.1f}s")
        st.audio(os_path)
        with open(os_path, "rb") as f:
            st.download_button("Download oneshot", f, file_name=os_name, mime="audio/wav", key=f"os_{i}")

    if all_files:
        zpath = os.path.join(out_dir, "sample_pack.zip")
        with zipfile.ZipFile(zpath, "w") as zf:
            for fpath in all_files:
                zf.write(fpath, Path(fpath).name)
        with open(zpath, "rb") as f:
            st.download_button("Download Full Pack", f, file_name="sample_pack.zip", key="pack")
