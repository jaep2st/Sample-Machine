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
st.caption("Best chops · cleaner loop alignment")

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

def apply_eq(seg, eq_type, amount):
    if eq_type == "None" or amount == 0:
        return seg
    if eq_type == "Low shelf":
        return seg * (1 + amount * 0.3) + np.convolve(seg, np.ones(9)/9, mode="same") * amount * 0.4
    if eq_type == "High shelf":
        return seg * (1 + amount * 0.2) + librosa.effects.preemphasis(seg, coef=0.9) * amount * 0.5
    return seg

def apply_crossfade_loop(seg, sr, crossfade_ms=30):
    if crossfade_ms <= 0 or len(seg) < sr * 0.2:
        return np.concatenate([seg] * 4)
    cf = int(crossfade_ms / 1000 * sr)
    cf = min(cf, len(seg)//4)
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

def norm(seg, gain=1.0):
    peak = np.max(np.abs(seg))
    if peak > 0:
        seg = seg / peak * 0.95 * gain
    return np.clip(seg, -1.0, 1.0)

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
            end_bonus = max(0.0, 1.2 - end_dist * 3)
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

def extract_oneshots(y, sr, max_shots=8):
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    shots = []
    for onset in onsets[:100]:
        window = y[int(onset*sr):int((onset+0.3)*sr)]
        if len(window) < 80:
            continue
        energy = float(np.mean(np.abs(window)))
        if energy > 0.07:
            peak = float(np.max(np.abs(window)))
            if peak > 0.18:
                shots.append((onset, energy * peak))
    shots.sort(key=lambda x: x[1], reverse=True)
    return shots[:max_shots]

uploaded = st.file_uploader("Drop a track", type=["mp3", "wav", "m4a", "flac"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded.getvalue())
        path = tmp.name

    y_full, sr = librosa.load(path, sr=44100, mono=True)
    duration = len(y_full) / sr
    key = detect_key(y_full, sr)
    st.write(f"Track length: {duration:.1f}s · **{key}**")

    lock_bpm = st.checkbox("Lock chops to target BPM", value=False)
    target_bpm = st.number_input("Target BPM", 60, 200, 90) if lock_bpm else None

    use_region = st.checkbox("Chop a specific section instead", value=False)
    if use_region:
        region_start = st.slider("Region start (sec)", 0.0, max(0.1, duration-1), 0.0, 0.1)
        region_end = st.slider("Region end (sec)", 1.0, duration, min(duration, 30.0), 0.1)
        if region_end > region_start:
            y_source = y_full[int(region_start*sr):int(region_end*sr)]
            region_offset = region_start
        else:
            y_source = y_full
            region_offset = 0.0
    else:
        y_source = y_full
        region_offset = 0.0

    source_key = f"{uploaded.name}_{region_offset}_{target_bpm}"
    if "auto_chops" not in st.session_state or st.session_state.get("source_key") != source_key:
        with st.spinner("Finding the best chops..."):
            selected, detected_bpm = find_best_chops(y_source, sr, bars=4, max_chops=6)
            bpm = target_bpm if lock_bpm else detected_bpm
            st.session_state.auto_chops = []
            st.session_state.bpm = bpm
            st.session_state.key = key
            st.session_state.source_key = source_key
            
            for i, (start, end, sc) in enumerate(selected, 1):
                st.session_state.auto_chops.append({
                    "id": i,
                    "score": sc,
                    "orig_start": start + region_offset,
                    "orig_end": end + region_offset,
                    "start": start + region_offset,
                    "end": end + region_offset,
                    "pitch": 0,
                    "reverse": False,
                    "style": "Clean",
                    "eq": "None",
                    "eq_amount": 0.5,
                    "fade_in": 8,
                    "fade_out": 8,
                    "crossfade": 20,
                    "volume": 1.0,
                    "include": True
                })

    bpm = st.session_state.bpm
    st.write(f"**{bpm:.1f} BPM** · **{key}**")

    out_dir = tempfile.mkdtemp()
    included_files = []
    oneshot_files = []

    for chop in st.session_state.auto_chops:
        st.markdown(f"### Chop {chop['id']} · Score: {chop['score']:.2f}")
        st.caption(f"Original: {chop['orig_start']:.2f}s – {chop['orig_end']:.2f}s")

        chop["include"] = st.checkbox("Include in pack", value=chop["include"], key=f"inc_{chop['id']}")

        # Cleaner trim controls
        col_start, col_end = st.columns(2)
        with col_start:
            start_trim = st.slider("Trim start (sec)", -0.5, 0.5, chop["start"] - chop["orig_start"], 0.01, key=f"st_{chop['id']}")
        with col_end:
            end_trim = st.slider("Trim end (sec)", -0.5, 0.5, chop["end"] - chop["orig_end"], 0.01, key=f"et_{chop['id']}")

        chop["start"] = max(0, chop["orig_start"] + start_trim)
        chop["end"] = min(duration, chop["orig_end"] + end_trim)
        if chop["end"] <= chop["start"] + 0.2:
            chop["end"] = chop["start"] + 0.2

        c1, c2, c3 = st.columns(3)
        with c1:
            chop["pitch"] = st.slider("Pitch", -12, 12, chop["pitch"], key=f"pitch_{chop['id']}")
        with c2:
            chop["reverse"] = st.checkbox("Reverse", value=chop["reverse"], key=f"rev_{chop['id']}")
        with c3:
            chop["style"] = st.selectbox("Style", ["Clean", "Dusty", "Warm", "Dark"],
                                         index=["Clean","Dusty","Warm","Dark"].index(chop["style"]),
                                         key=f"style_{chop['id']}")

        c4, c5 = st.columns(2)
        with c4:
            chop["eq"] = st.selectbox("EQ", ["None", "Low shelf", "High shelf"],
                                      index=["None","Low shelf","High shelf"].index(chop["eq"]),
                                      key=f"eq_{chop['id']}")
        with c5:
            if chop["eq"] != "None":
                chop["eq_amount"] = st.slider("EQ amount", 0.0, 1.0, chop["eq_amount"], 0.05, key=f"eqamt_{chop['id']}")

        c6, c7 = st.columns(2)
        with c6:
            chop["fade_in"] = st.slider("Fade in (ms)", 0, 80, chop["fade_in"], 2, key=f"fi_{chop['id']}")
        with c7:
            chop["fade_out"] = st.slider("Fade out (ms)", 0, 80, chop["fade_out"], 2, key=f"fo_{chop['id']}")

        chop["volume"] = st.slider("Volume", 0.5, 1.5, chop["volume"], 0.05, key=f"vol_{chop['id']}")
        chop["crossfade"] = st.slider("Loop crossfade (ms)", 0, 80, chop["crossfade"], 2, key=f"cf_{chop['id']}")

        start_sample = int(max(0, chop["start"]) * sr)
        end_sample = int(min(chop["end"], duration) * sr)
        seg = y_full[start_sample:end_sample].copy()

        if chop["pitch"] != 0:
            seg = librosa.effects.pitch_shift(seg, sr=sr, n_steps=chop["pitch"])
        if chop["reverse"]:
            seg = seg[::-1]
        seg = process(seg, chop["style"])
        seg = apply_eq(seg, chop["eq"], chop["eq_amount"])
        seg = apply_fade(seg, chop["fade_in"], chop["fade_out"], sr)
        seg = norm(seg, chop["volume"])

        name = f"{int(bpm)}bpm_{key}_chop{chop['id']:02d}"
        if chop["pitch"] != 0:
            name += f"_p{chop['pitch']:+d}"
        if chop["reverse"]:
            name += "_rev"
        if chop["style"] != "Clean":
            name += f"_{chop['style'].lower()}"
        if chop["eq"] != "None":
            name += f"_{chop['eq'].lower().replace(' ','')}"
        name += ".wav"

        tmp_path = os.path.join(out_dir, name)
        sf.write(tmp_path, seg, sr)
        if chop["include"]:
            included_files.append(tmp_path)

        st.markdown("**Single**")
        st.audio(tmp_path)
        with open(tmp_path, "rb") as f:
            st.download_button("Download single", f, file_name=name, mime="audio/wav", key=f"dl_{chop['id']}")

        loop_seg = seg[::-1] if chop["reverse"] else seg
        looped = apply_crossfade_loop(loop_seg, sr, chop["crossfade"])
        loop_name = name.replace(".wav", "_loop.wav")
        loop_path = os.path.join(out_dir, loop_name)
        sf.write(loop_path, looped, sr)
        if chop["include"]:
            included_files.append(loop_path)
        st.markdown(f"**Loop (4x, {chop['crossfade']}ms crossfade)**")
        st.audio(loop_path)

    # A/B
    st.divider()
    st.subheader("Compare chops")
    if len(st.session_state.auto_chops) >= 2:
        ids = [c["id"] for c in st.session_state.auto_chops]
        a = st.selectbox("Chop A", ids, key="cmp_a")
        b = st.selectbox("Chop B", ids, index=min(1, len(ids)-1), key="cmp_b")
        if st.button("Play A then B"):
            st.success(f"Playing Chop {a} then Chop {b}")

    # One-shots
    st.divider()
    st.subheader("One-shots")
    max_oneshots = st.slider("One-shots to extract", 4, 16, 8)
    oneshots = extract_oneshots(y_full, sr, max_shots=max_oneshots)
    include_oneshots = st.checkbox("Include one-shots in pack", value=True)

    for i, (onset, energy) in enumerate(oneshots, 1):
        window = y_full[int(onset*sr):int((onset+0.45)*sr)]
        window = apply_fade(window, 5, 5, sr)
        window = norm(window)
        os_name = f"{int(bpm)}bpm_{key}_oneshot{i:02d}_{onset:.1f}s.wav"
        os_path = os.path.join(out_dir, os_name)
        sf.write(os_path, window, sr)
        if include_oneshots:
            oneshot_files.append(os_path)
        st.markdown(f"**One-shot {i}** · {onset:.1f}s")
        st.audio(os_path)
        with open(os_path, "rb") as f:
            st.download_button("Download oneshot", f, file_name=os_name, mime="audio/wav", key=f"os_{i}")

    pack_files = included_files + oneshot_files
    if pack_files:
        zpath = os.path.join(out_dir, "sample_pack.zip")
        with zipfile.ZipFile(zpath, "w") as zf:
            for fpath in pack_files:
                zf.write(fpath, Path(fpath).name)
        with open(zpath, "rb") as f:
            st.download_button("Download Selected Pack", f, file_name="sample_pack.zip", key="pack")
