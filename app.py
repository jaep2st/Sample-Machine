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
st.caption("Target, refine, and lock chops")

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

    c1, c2, c3 = st.columns(3)
    with c1:
        mode = st.selectbox("Mode", ["Phrase", "Micro"])
    with c2:
        pitch = st.slider("Pitch", -12, 12, 0)
    with c3:
        style = st.selectbox("Style", ["Clean", "Dusty", "Warm", "Dark"])

    if mode == "Phrase":
        bars = st.slider("Bars", 1, 8, 4)
    else:
        bars = st.slider("Chop length (beats)", 1, 4, 2)

    max_chops = st.slider("Chops", 1, 8, 3)
    fade_ms = st.slider("Fade (ms)", 0, 100, 10)
    oneshot_len = st.slider("One-shot length (sec)", 0.5, 3.0, 1.5, 0.1)
    humanize = st.checkbox("Humanize starts", value=True)
    normalize = st.checkbox("Normalize", value=True)

    if "chops" not in st.session_state:
        st.session_state.chops = []

    if st.button("Chop this section", type="primary"):
        y = y_region.copy()

        # Pitch first
        if pitch != 0:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        beat_times = librosa.frames_to_time(beats, sr=sr)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        key = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][int(np.argmax(np.mean(chroma, axis=1)))]

        onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
        rms = librosa.feature.rms(y=y)[0]
        t = librosa.times_like(rms, sr=sr)

        if mode == "Phrase":
            bar_dur = (bars * 4 * 60) / bpm
        else:
            bar_dur = (bars * 60) / bpm   # shorter micro chops

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

        def apply_fade(seg, fade_ms, sr):
            if fade_ms <= 0 or len(seg) == 0:
                return seg
            fade_samples = int(fade_ms / 1000 * sr)
            fade_samples = min(fade_samples, len(seg)//4)
            if fade_samples > 0:
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                seg = seg.copy()
                seg[:fade_samples] *= fade_in
                seg[-fade_samples:] *= fade_out
            return seg

        def norm(seg):
            if not normalize or len(seg) == 0:
                return seg
            peak = np.max(np.abs(seg))
            return seg / peak * 0.95 if peak > 0 else seg

        st.session_state.chops = []
        st.session_state.meta = {
            "bpm": bpm, "key": key, "y": y, "sr": sr,
            "region_start": region_start, "style": style, "pitch": pitch, "mode": mode
        }

        for i, (start, end, _) in enumerate(selected, 1):
            # Humanize
            if humanize:
                offset = np.random.uniform(-0.03, 0.03)
                start = max(0, start + offset)
                end = start + bar_dur

            seg = y[int(start*sr):int(end*sr)].copy()
            seg = process(seg, style)
            seg = apply_fade(seg, fade_ms, sr)
            seg = norm(seg)

            abs_start = region_start + start
            base = f"{int(bpm)}bpm_{key}_{style.lower()}_{mode.lower()}_{i:02d}_{abs_start:.1f}s"
            if pitch != 0:
                base += f"_p{pitch:+d}"

            st.session_state.chops.append({
                "base": base,
                "loop": seg,
                "start": start,
                "end": end,
                "locked": False
            })

    if st.session_state.chops:
        meta = st.session_state.meta
        st.write(f"**{meta['bpm']:.1f} BPM** · **{meta['key']}** · {meta['mode']}")
        st.success(f"{len(st.session_state.chops)} chops")

        out = tempfile.mkdtemp()
        files = []

        for idx, chop in enumerate(st.session_state.chops):
            col1, col2, col3 = st.columns([4, 1, 1])
            with col1:
                st.markdown(f"**{chop['base']}**")
            with col2:
                chop["locked"] = st.checkbox("Lock", value=chop["locked"], key=f"lock_{idx}")
            with col3:
                if not chop["locked"] and st.button("Regen", key=f"regen_{idx}"):
                    y = meta["y"]
                    new_start = max(0, chop["start"] + np.random.uniform(-0.4, 0.4))
                    new_end = new_start + (chop["end"] - chop["start"])
                    if new_end < len(y)/meta["sr"]:
                        seg = y[int(new_start*meta["sr"]):int(new_end*meta["sr"])].copy()
                        seg = process(seg, meta["style"])
                        seg = apply_fade(seg, fade_ms, meta["sr"])
                        seg = norm(seg)
                        chop["loop"] = seg
                        chop["start"] = new_start
                        st.rerun()

            tmp_loop = os.path.join(out, f"tmp_{idx}_loop.wav")
            sf.write(tmp_loop, chop["loop"], meta["sr"])
            st.audio(tmp_loop)

            env = np.abs(chop["loop"])
            if len(env) > 10:
                peak_idx = np.argmax(env[:int(meta["sr"]*0.4)])
                os_len = int(oneshot_len * meta["sr"])
                oneshot = chop["loop"][max(0, peak_idx-int(meta["sr"]*0.02)):peak_idx+os_len]
                oneshot = norm(oneshot)
                tmp_os = os.path.join(out, f"tmp_{idx}_os.wav")
                sf.write(tmp_os, oneshot, meta["sr"])
                st.audio(tmp_os)

            loop_path = os.path.join(out, chop["base"] + "_loop.wav")
            sf.write(loop_path, chop["loop"], meta["sr"])
            files.append(loop_path)
            with open(loop_path, "rb") as fh:
                st.download_button("Save loop", fh, file_name=Path(loop_path).name, mime="audio/wav", key=f"dl_loop_{idx}")

        if files:
            zpath = os.path.join(out, "pack.zip")
            with zipfile.ZipFile(zpath, "w") as zf:
                for fpath in files:
                    zf.write(fpath, Path(fpath).name)
            with open(zpath, "rb") as fh:
                st.download_button("Download Pack", fh, file_name="sample_pack.zip", key="zip")
