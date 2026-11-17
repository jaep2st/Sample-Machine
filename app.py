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
st.caption("Best · Weird · All · One-shots")

def process(seg, style):
    if style == "Dusty":
        seg = np.convolve(seg, np.ones(18)/18, mode="same")
        seg = np.tanh(seg * 1.4) * 0.82
    elif style == "Warm":
        seg = np.convolve(seg, np.ones(6)/6, mode="same")
        seg = np.tanh(seg * 1.12) * 0.94
    elif style == "Dark":
        seg = np.convolve(seg, np.ones(28)/28, mode="same")
        seg = seg * 0.62
    elif style == "Tape":
        seg = np.convolve(seg, np.ones(9)/9, mode="same")
        seg = np.tanh(seg * 1.25) * 0.88
    return seg

def apply_crossfade_loop(seg, sr, crossfade_ms=25):
    if len(seg) < sr * 0.2:
        return np.concatenate([seg] * 4)
    cf = int(crossfade_ms / 1000 * sr)
    cf = min(cf, len(seg)//6)
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

def snap_to_beat(time, beat_times):
    if len(beat_times) == 0:
        return time
    return beat_times[np.argmin(np.abs(beat_times - time))]

def score_chop(y, sr, start, end):
    seg = y[int(start*sr):int(end*sr)]
    if len(seg) < 100:
        return 0.0
    rms = float(np.mean(np.abs(seg)))
    balance = 1.0 - abs(rms - 0.12) * 3
    return max(0.0, rms * balance)

def chop_whole_song(y, sr, bars=2):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)
    song_len = len(y) / sr
    bar_len = (bars * 4 * 60) / bpm
    
    start = snap_to_beat(0.5, beat_times)
    chops = []
    i = 1
    while start + bar_len < song_len - 0.1:
        end = snap_to_beat(start + bar_len, beat_times)
        if end <= start + 0.2:
            end = start + bar_len
        score = score_chop(y, sr, start, end)
        chops.append((start, end, i, score))
        start = end
        i += 1
    return chops, bpm, beat_times

def extract_oneshots(y, sr, max_shots=12):
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True, delta=0.05)
    shots = []
    for onset in onsets[:200]:
        window = y[int(onset*sr):int((onset+0.8)*sr)]
        if len(window) < 40:
            continue
        rms = np.abs(window)
        peak = float(np.max(rms))
        cutoff = int(onset*sr)
        for j, val in enumerate(rms):
            if j > int(0.1*sr) and val < peak * 0.3:
                cutoff = int(onset*sr) + j
                break
        length = min(0.8, (cutoff - int(onset*sr)) / sr)
        if length < 0.15:
            length = 0.3
        hit = y[int(onset*sr):int((onset+0.08)*sr)]
        hit_energy = float(np.mean(np.abs(hit))) if len(hit) else 0
        if hit_energy > 0.08 and peak > 0.25:
            shots.append((onset, length, hit_energy * peak))
    shots.sort(key=lambda x: x[2], reverse=True)
    return shots[:max_shots]

if "hearted" not in st.session_state:
    st.session_state.hearted = set()

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

    with st.spinner("Chopping the whole song..."):
        all_chops, bpm, beat_times = chop_whole_song(y_full, sr, bars=bars)

    scored = sorted(all_chops, key=lambda x: x[3], reverse=True)
    best_chops = scored[:max(6, len(scored)//3)]
    weird_chops = [c for c in scored if c not in best_chops][:8]

    tab_best, tab_weird, tab_all, tab_oneshots = st.tabs(["Best", "Weird", "All", "One-shots"])

    out_dir = tempfile.mkdtemp()
    included = []
    loops_only = []
    hearted_files = []

    def render_chop(start, end, i, label="Chop", tab="all"):
        chop_key = f"chop_{i}"
        wk = f"{tab}_{i}"
        col1, col2 = st.columns([8, 1])
        with col1:
            st.markdown(f"**{label} {i}** · from {start:.1f}s")
        with col2:
            hearted = chop_key in st.session_state.hearted
            if st.button("♥" if hearted else "♡", key=f"heart_{wk}"):
                if hearted:
                    st.session_state.hearted.discard(chop_key)
                else:
                    st.session_state.hearted.add(chop_key)
                st.rerun()

        shift = 0.0
        trim_start = 0.0
        trim_end = 0.0
        pitch = 0
        rev = False
        style = "Clean"

        with st.expander("Tweak this chop", expanded=False):
            shift = st.slider("Shift whole chop", -2.0, 2.0, 0.0, 0.05, key=f"shift_{wk}")
            trim_start = st.slider("Trim start", -0.5, 0.5, 0.0, 0.01, key=f"ts_{wk}")
            trim_end = st.slider("Trim end", -0.5, 0.5, 0.0, 0.01, key=f"te_{wk}")
            pitch = st.slider("Pitch", -12, 12, 0, key=f"p_{wk}")
            rev = st.checkbox("Reverse", key=f"r_{wk}")
            style = st.selectbox("Style", ["Clean", "Dusty", "Warm", "Dark", "Tape"], key=f"s_{wk}")

        # Shift moves the whole window, then trim adjusts the edges
        adj_start = max(0, start + shift + trim_start)
        adj_end = min(duration, end + shift + trim_end)
        if adj_end <= adj_start + 0.2:
            adj_end = adj_start + 0.2

        seg = y_full[int(adj_start*sr):int(adj_end*sr)].copy()
        if pitch != 0:
            seg = librosa.effects.pitch_shift(seg, sr=sr, n_steps=pitch)
        if rev:
            seg = do_reverse(seg, sr)
        seg = process(seg, style)
        seg = norm(seg)

        name = f"{int(bpm)}bpm_{key}_chop{i:02d}_{adj_start:.1f}s"
        if pitch != 0:
            name += f"_p{pitch:+d}"
        if rev:
            name += "_rev"
        if style != "Clean":
            name += f"_{style.lower()}"
        name += ".wav"

        tmp_path = os.path.join(out_dir, name)
        sf.write(tmp_path, seg, sr)
        included.append(tmp_path)
        if chop_key in st.session_state.hearted:
            hearted_files.append(tmp_path)

        looped = apply_crossfade_loop(seg, sr, crossfade_ms=25)
        loop_path = os.path.join(out_dir, name.replace(".wav", "_loop.wav"))
        sf.write(loop_path, looped, sr)
        loops_only.append(loop_path)
        if chop_key in st.session_state.hearted:
            hearted_files.append(loop_path)

        st.audio(loop_path)
        st.audio(tmp_path)
        with open(tmp_path, "rb") as f:
            st.download_button("Download", f, file_name=name, mime="audio/wav", key=f"dl_{wk}")

    with tab_best:
        st.write(f"**{bpm:.1f} BPM** · {len(best_chops)} best chops")
        for start, end, i, score in best_chops:
            render_chop(start, end, i, "Best", "best")

    with tab_weird:
        st.write(f"**{bpm:.1f} BPM** · {len(weird_chops)} weird chops")
        for start, end, i, score in weird_chops:
            render_chop(start, end, i, "Weird", "weird")

    with tab_all:
        st.write(f"**{bpm:.1f} BPM** · {len(all_chops)} chops")
        for start, end, i, score in all_chops:
            render_chop(start, end, i, "Chop", "all")

    with tab_oneshots:
        oneshots = extract_oneshots(y_full, sr)
        st.write(f"{len(oneshots)} one-shots")
        os_files = []
        for i, (onset, length, energy) in enumerate(oneshots, 1):
            os_key = f"os_{i}"
            col1, col2 = st.columns([8, 1])
            with col1:
                st.markdown(f"**One-shot {i}** · from {onset:.1f}s · {length:.2f}s")
            with col2:
                hearted = os_key in st.session_state.hearted
                if st.button("♥" if hearted else "♡", key=f"heartos_{i}"):
                    if hearted:
                        st.session_state.hearted.discard(os_key)
                    else:
                        st.session_state.hearted.add(os_key)
                    st.rerun()

            window = y_full[int(onset*sr):int((onset+length)*sr)].copy()
            window = norm(window)
            os_name = f"{int(bpm)}bpm_{key}_oneshot{i:02d}_{onset:.1f}s.wav"
            os_path = os.path.join(out_dir, os_name)
            sf.write(os_path, window, sr)
            os_files.append(os_path)
            if os_key in st.session_state.hearted:
                hearted_files.append(os_path)

            loop_os = np.concatenate([window] * 4)
            loop_os_path = os.path.join(out_dir, f"osloop_{i}.wav")
            sf.write(loop_os_path, loop_os, sr)
            if os_key in st.session_state.hearted:
                hearted_files.append(loop_os_path)

            st.audio(loop_os_path)
            st.audio(os_path)
            with open(os_path, "rb") as f:
                st.download_button("Download", f, file_name=os_name, mime="audio/wav", key=f"osdl_{i}")

        if os_files:
            zpath = os.path.join(out_dir, "oneshots.zip")
            with zipfile.ZipFile(zpath, "w") as zf:
                for fpath in os_files:
                    zf.write(fpath, Path(fpath).name)
            with open(zpath, "rb") as f:
                st.download_button("Download All One-shots", f, file_name="oneshots.zip", key="allos")

    if hearted_files:
        zpath = os.path.join(out_dir, "favorites.zip")
        with zipfile.ZipFile(zpath, "w") as zf:
            for fpath in hearted_files:
                zf.write(fpath, Path(fpath).name)
        with open(zpath, "rb") as f:
            st.download_button("Download Favorites", f, file_name="favorites.zip", key="favpack")

    if loops_only:
        zpath = os.path.join(out_dir, "loops_only.zip")
        with zipfile.ZipFile(zpath, "w") as zf:
            for fpath in loops_only:
                zf.write(fpath, Path(fpath).name)
        with open(zpath, "rb") as f:
            st.download_button("Download Loops Only", f, file_name="loops_only.zip", key="loopsonly")

    if included:
        zpath = os.path.join(out_dir, "all_chops.zip")
        with zipfile.ZipFile(zpath, "w") as zf:
            for fpath in included:
                zf.write(fpath, Path(fpath).name)
        with open(zpath, "rb") as f:
            st.download_button("Download All Chops", f, file_name="all_chops.zip", key="allchops")
