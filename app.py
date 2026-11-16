import streamlit as st
import os
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import librosa
from pydub import AudioSegment

st.set_page_config(page_title="Hip Hop Sample Machine", page_icon="🎧", layout="centered")

st.title("🎧 Hip Hop Sample Machine")
st.caption("Upload a song → get 8-bar chops (original, pitched, filtered, reversed)")

uploaded = st.file_uploader("Drop an audio file (MP3, WAV, M4A)", type=["mp3", "wav", "m4a", "flac"])

if uploaded:
    with st.spinner("Analyzing and chopping..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.getvalue())
            audio_path = tmp.name

        y, sr = librosa.load(audio_path, sr=44100, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        st.metric("BPM", f"{bpm:.1f}")

        # 8-bar regions
        bar_duration = (8 * 4 * 60) / bpm
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units="time")
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

        candidates = []
        for start in onsets:
            end = start + bar_duration
            if end > len(y) / sr:
                continue
            mask = (times >= start) & (times <= end)
            if np.any(mask):
                energy = np.mean(rms[mask])
                candidates.append((start, end, energy))

        candidates.sort(key=lambda x: x[2], reverse=True)
        selected = []
        for start, end, energy in candidates:
            if all(not (start < s_end and end > s_start) for s_start, s_end, _ in selected):
                selected.append((start, end, energy))
            if len(selected) >= 8:
                break

        full_audio = AudioSegment.from_file(audio_path)
        out_dir = tempfile.mkdtemp()
        samples = []

        for idx, (start, end, energy) in enumerate(selected, 1):
            segment = full_audio[int(start * 1000):int(end * 1000)]
            base = f"chop_{idx:02d}_8bars"

            # Original
            p = os.path.join(out_dir, f"{base}_original.wav")
            segment.export(p, format="wav")
            samples.append(p)

            # Pitched down
            pitched = segment._spawn(
                segment.raw_data,
                overrides={"frame_rate": int(segment.frame_rate * (2 ** (-3 / 12)))},
            ).set_frame_rate(segment.frame_rate)
            p = os.path.join(out_dir, f"{base}_pitched_down.wav")
            pitched.export(p, format="wav")
            samples.append(p)

            # Filtered
            filtered = segment.low_pass_filter(8000)
            p = os.path.join(out_dir, f"{base}_filtered.wav")
            filtered.export(p, format="wav")
            samples.append(p)

            # Reversed
            rev = segment.reverse()
            p = os.path.join(out_dir, f"{base}_reversed.wav")
            rev.export(p, format="wav")
            samples.append(p)

    st.success(f"Found {len(selected)} strong 8-bar chops")

    # Preview + individual download
    for s in samples:
        name = Path(s).name
        st.subheader(name)
        st.audio(s)
        with open(s, "rb") as f:
            st.download_button("Download this one", f, file_name=name, mime="audio/wav", key=name)

    # Full ZIP
    zip_path = os.path.join(out_dir, "hiphop_sample_pack.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for s in samples:
            zf.write(s, Path(s).name)

    with open(zip_path, "rb") as f:
        st.download_button(
            "📦 Download Full Pack (ZIP)",
            f,
            file_name="hiphop_sample_pack.zip",
            mime="application/zip",
        )
