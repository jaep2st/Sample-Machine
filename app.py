import streamlit as st
import os
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Sample Machine", page_icon="🎧", layout="centered")
st.title("Sample Machine")
st.caption("Best · Weird · All · One-shots")

KEYS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

try:
    from supabase import create_client
    SUPABASE_URL = st.secrets["SUPABASE_URL"]
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception:
    supabase = None

def get_user():
    if not supabase:
        return None
    try:
        return supabase.auth.get_user()
    except Exception:
        return None

def toggle_heart(user_id, item_key):
    if not supabase or not user_id:
        return
    try:
        existing = supabase.table("hearts").select("id").eq("user_id", user_id).eq("item_key", item_key).execute()
        if existing.data:
            supabase.table("hearts").delete().eq("user_id", user_id).eq("item_key", item_key).execute()
        else:
            supabase.table("hearts").insert({"user_id": user_id, "item_key": item_key}).execute()
    except Exception:
        pass

def get_hearts(user_id):
    if not supabase or not user_id:
        return set()
    try:
        res = supabase.table("hearts").select("item_key").eq("user_id", user_id).execute()
        return {row["item_key"] for row in res.data}
    except Exception:
        return set()

def process(seg, style):
    if len(seg) < 100:
        return seg
    if style == "Dusty":
        seg = np.convolve(seg, np.ones(16)/16, mode="same")
        seg = np.tanh(seg * 1.35) * 0.84
    elif style == "Warm":
        low = np.convolve(seg, np.ones(7)/7, mode="same")
        seg = seg * 0.7 + low * 0.4
        seg = np.tanh(seg * 1.1) * 0.95
    elif style == "Dark":
        seg = np.convolve(seg, np.ones(32)/32, mode="same")
        seg = seg * 0.6
    elif style == "Tape":
        seg = np.convolve(seg, np.ones(8)/8, mode="same")
        seg = np.tanh(seg * 1.28) * 0.87
    elif style == "Crushed":
        seg = np.round(seg * 12) / 12
        seg = np.clip(seg * 1.4, -0.95, 0.95)
    elif style == "Filtered":
        low = np.convolve(seg, np.ones(20)/20, mode="same")
        high = seg - np.convolve(seg, np.ones(4)/4, mode="same")
        seg = low * 0.5 + high * 0.6
        seg = np.tanh(seg * 1.15) * 0.9
    elif style == "Lo-fi":
        seg = np.convolve(seg, np.ones(12)/12, mode="same")
        noise = np.random.uniform(-0.008, 0.008, len(seg))
        seg = np.tanh((seg + noise) * 1.2) * 0.86
    elif style == "Chorus":
        delay = int(0.012 * 44100)
        delayed = np.pad(seg, (delay, 0))[:len(seg)]
        seg = seg * 0.7 + delayed * 0.35
        seg = np.tanh(seg * 1.05) * 0.93
    return seg

def apply_crossfade_loop(seg, sr, crossfade_ms=12):
    if len(seg) < sr * 0.08:
        return np.tile(seg, 4) if len(seg) > 0 else np.zeros(int(sr * 0.5))
    # Prefer zero-crossing near the end
    search = int(0.02 * sr)
    search = min(search, len(seg)//5)
    tail = seg[-search:]
    zc = np.where(np.diff(np.sign(tail)))[0]
    if len(zc) > 0:
        cut = len(seg) - search + zc[-1]
        if cut > len(seg) * 0.7:
            seg = seg[:cut]
    if len(seg) < 100:
        return np.tile(seg, 4)
    cf = int(crossfade_ms / 1000 * sr)
    cf = min(cf, max(1, len(seg)//10))
    fade_out = np.linspace(1, 0, cf)
    fade_in = np.linspace(0, 1, cf)
    looped = []
    for i in range(4):
        chunk = seg.copy()
        if i > 0 and cf < len(chunk):
            chunk[:cf] = chunk[:cf] * fade_in + seg[-cf:] * fade_out
        looped.append(chunk)
    result = np.concatenate(looped)
    if np.any(np.isnan(result)) or np.any(np.isinf(result)):
        return np.tile(seg, 4)
    return result

def do_reverse(seg, sr):
    return np.flip(seg).copy() if len(seg) > 0 else seg

def norm(seg, gain=1.0):
    if len(seg) == 0:
        return seg
    peak = np.max(np.abs(seg))
    if peak > 0:
        seg = seg / peak * 0.92 * gain
    return np.clip(seg, -1.0, 1.0)

def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean_chroma = np.mean(chroma, axis=1)
    major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
    best_score = -1
    best_key = "C"
    best_mode = "major"
    for i in range(12):
        rotated_major = np.roll(major_template, i)
        rotated_minor = np.roll(minor_template, i)
        major_score = np.dot(mean_chroma, rotated_major)
        minor_score = np.dot(mean_chroma, rotated_minor)
        if major_score > best_score:
            best_score = major_score
            best_key = KEYS[i]
            best_mode = "major"
        if minor_score > best_score:
            best_score = minor_score
            best_key = KEYS[i]
            best_mode = "minor"
    return f"{best_key} {best_mode}", KEYS.index(best_key)

def shift_key_name(key_str, semitones):
    if semitones == 0:
        return key_str
    parts = key_str.split()
    root = parts[0]
    mode = parts[1] if len(parts) > 1 else "major"
    idx = KEYS.index(root)
    new_idx = (idx + semitones) % 12
    return f"{KEYS[new_idx]} {mode}"

def snap_to_beat(time, beat_times):
    if len(beat_times) == 0:
        return time
    return beat_times[np.argmin(np.abs(beat_times - time))]

def score_chop(y, sr, start, end, song_len):
    seg = y[int(start*sr):int(end*sr)]
    if len(seg) < 100:
        return 0.0
    rms = float(np.mean(np.abs(seg)))
    energy_score = 1.0 - abs(rms - 0.11) * 4
    energy_score = max(0.0, energy_score)
    diff = np.abs(np.diff(seg))
    transient = float(np.mean(diff[:min(len(diff), int(0.1*sr))]))
    transient_score = min(1.0, transient * 8)
    position_score = 1.0 - (start / song_len) * 0.4
    start_sample = int(start * sr)
    window = y[max(0, start_sample-int(0.05*sr)):start_sample+int(0.1*sr)]
    onset_strength = float(np.mean(np.abs(window))) if len(window) else 0
    onset_score = min(1.0, onset_strength * 3)
    score = (energy_score * 0.4) + (transient_score * 0.25) + (position_score * 0.2) + (onset_score * 0.15)
    return max(0.0, score)

def estimate_bpm(y, sr):
    # Primary method
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, start_bpm=100)
    bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    # Fallback if BPM is extreme
    if bpm < 50 or bpm > 180:
        tempo2, _ = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120)
        bpm2 = float(tempo2) if np.isscalar(tempo2) else float(tempo2[0])
        if 60 <= bpm2 <= 160:
            bpm = bpm2
    # Half/double correction for common doubling
    if bpm > 140:
        bpm = bpm / 2
    elif bpm < 70:
        bpm = bpm * 2
    return max(60.0, min(180.0, bpm)), beats

def chop_whole_song(y, sr, bars=2):
    bpm, beats = estimate_bpm(y, sr)
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
        score = score_chop(y, sr, start, end, song_len)
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

@st.cache_data(show_spinner=False)
def make_waveform_image(y_bytes, sr, start, end, width=500, height=28):
    y = np.frombuffer(y_bytes, dtype=np.float32)
    hop = max(1, len(y) // 500)
    ys = y[::hop]
    t = np.linspace(0, len(y)/sr, len(ys))
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=70)
    ax.plot(t, ys, color="#888888", linewidth=0.25)
    ax.axvspan(start, end, color="#4CAF50", alpha=0.4)
    ax.set_xlim(0, len(y)/sr)
    ax.set_ylim(-1, 1)
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# Auth at top
if supabase:
    user = get_user()
    if user is None:
        with st.expander("Log in or sign up", expanded=False):
            mode = st.radio(" ", ["Log in", "Sign up"], horizontal=True, label_visibility="collapsed")
            email = st.text_input("Email", key="auth_email")
            password = st.text_input("Password", type="password", key="auth_pass")
            if mode == "Log in":
                if st.button("Log in", use_container_width=True):
                    try:
                        supabase.auth.sign_in_with_password({"email": email, "password": password})
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
            else:
                if st.button("Sign up", use_container_width=True):
                    try:
                        supabase.auth.sign_up({"email": email, "password": password})
                        st.success("Check your email to confirm.")
                    except Exception as e:
                        st.error(str(e))
    else:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.caption(f"Logged in as {user.user.email}")
        with col2:
            if st.button("Log out", use_container_width=True):
                supabase.auth.sign_out()
                st.rerun()

if "hearted" not in st.session_state:
    st.session_state.hearted = set()

if supabase:
    user = get_user()
    if user:
        st.session_state.hearted = get_hearts(user.user.id)

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
    key, key_idx = detect_key(y_full, sr)
    st.write(f"Track length: {duration:.1f}s · **{key}**")

    bar_choice = st.radio("Bar length", ["2 bars", "4 bars", "8 bars", "16 bars"], index=0, horizontal=True)
    bars = int(bar_choice.split()[0])

    with st.spinner("Chopping the whole song..."):
        all_chops, bpm, beat_times = chop_whole_song(y_full, sr, bars=bars)

    scored = sorted(all_chops, key=lambda x: x[3], reverse=True)
    best_chops = scored[:max(6, len(scored)//3)]
    weird_chops = [c for c in scored if c not in best_chops][:8]

    y_bytes = y_full.astype(np.float32).tobytes()

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
            st.markdown(f"**{label} {i}** · {int(bpm)} BPM · {key} · {start:.1f}s")
        with col2:
            hearted = chop_key in st.session_state.hearted
            if st.button("♥" if hearted else "♡", key=f"heart_{wk}"):
                user = get_user()
                if user:
                    toggle_heart(user.user.id, chop_key)
                    st.session_state.hearted = get_hearts(user.user.id)
                else:
                    if hearted:
                        st.session_state.hearted.discard(chop_key)
                    else:
                        st.session_state.hearted.add(chop_key)
                st.rerun()

        try:
            img = make_waveform_image(y_bytes, sr, start, end)
            st.image(img, use_container_width=True)
        except Exception:
            pass

        shift = 0.0
        trim_start = 0.0
        trim_end = 0.0
        pitch = 0
        rev = False
        style = "Clean"
        new_bpm = int(bpm)

        with st.expander("Tweak this chop", expanded=False):
            shift = st.slider("Shift whole chop", -2.0, 2.0, 0.0, 0.05, key=f"shift_{wk}")
            trim_start = st.slider("Trim start", -0.5, 0.5, 0.0, 0.01, key=f"ts_{wk}")
            trim_end = st.slider("Trim end", -0.5, 0.5, 0.0, 0.01, key=f"te_{wk}")
            pitch = st.slider("Pitch", -12, 12, 0, key=f"p_{wk}")
            new_bpm = st.slider("BPM", max(40, int(bpm)-40), min(200, int(bpm)+40), int(bpm), key=f"bpm_{wk}")
            rev = st.checkbox("Reverse", key=f"r_{wk}")
            style = st.selectbox("Style", ["Clean", "Dusty", "Warm", "Dark", "Tape", "Crushed", "Filtered", "Lo-fi", "Chorus"], key=f"s_{wk}")

        raw_start = start + shift + trim_start
        raw_end = end + shift + trim_end
        adj_start = max(0.0, min(raw_start, duration - 0.25))
        adj_end = max(adj_start + 0.25, min(raw_end, duration))

        seg = y_full[int(adj_start*sr):int(adj_end*sr)].copy()
        if len(seg) < 100:
            seg = np.zeros(int(sr * 0.5))

        new_key = key
        if pitch != 0 and len(seg) > 100:
            try:
                seg = librosa.effects.pitch_shift(seg, sr=sr, n_steps=pitch, bins_per_octave=36)
                new_key = shift_key_name(key, pitch)
            except Exception:
                pass

        if new_bpm != int(bpm) and len(seg) > 100:
            try:
                rate = bpm / new_bpm
                seg = librosa.effects.time_stretch(seg, rate=rate)
            except Exception:
                pass

        if rev and len(seg) > 0:
            seg = do_reverse(seg, sr)
        seg = process(seg, style)
        seg = norm(seg)

        name = f"{new_bpm}bpm_{new_key.replace(' ', '_')}_chop{i:02d}_{adj_start:.1f}s"
        if pitch != 0:
            name += f"_p{pitch:+d}"
        if rev:
            name += "_rev"
        if style != "Clean":
            name += f"_{style.lower()}"
        name += ".wav"

        if pitch != 0 or new_bpm != int(bpm):
            changes = []
            if pitch != 0:
                changes.append(f"**{new_key}**")
            if new_bpm != int(bpm):
                changes.append(f"**{new_bpm} BPM**")
            st.caption(" · ".join(changes))

        tmp_path = os.path.join(out_dir, name)
        try:
            sf.write(tmp_path, seg, sr)
        except Exception:
            sf.write(tmp_path, np.zeros(int(sr * 0.5)), sr)
        included.append(tmp_path)
        if chop_key in st.session_state.hearted:
            hearted_files.append(tmp_path)

        looped = apply_crossfade_loop(seg, sr, crossfade_ms=12)
        loop_path = os.path.join(out_dir, name.replace(".wav", "_loop.wav"))
        try:
            sf.write(loop_path, looped, sr)
        except Exception:
            sf.write(loop_path, np.tile(seg, 4) if len(seg) > 0 else np.zeros(int(sr * 0.5)), sr)
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
                st.markdown(f"**One-shot {i}** · {int(bpm)} BPM · {key} · {onset:.1f}s")
            with col2:
                hearted = os_key in st.session_state.hearted
                if st.button("♥" if hearted else "♡", key=f"heartos_{i}"):
                    user = get_user()
                    if user:
                        toggle_heart(user.user.id, os_key)
                        st.session_state.hearted = get_hearts(user.user.id)
                    else:
                        if hearted:
                            st.session_state.hearted.discard(os_key)
                        else:
                            st.session_state.hearted.add(os_key)
                    st.rerun()

            try:
                img = make_waveform_image(y_bytes, sr, onset, onset + length)
                st.image(img, use_container_width=True)
            except Exception:
                pass

            window = y_full[int(onset*sr):int((onset+length)*sr)].copy()
            window = norm(window)
            os_name = f"{int(bpm)}bpm_{key.replace(' ', '_')}_oneshot{i:02d}_{onset:.1f}s.wav"
            os_path = os.path.join(out_dir, os_name)
            try:
                sf.write(os_path, window, sr)
            except Exception:
                sf.write(os_path, np.zeros(int(sr * 0.5)), sr)
            os_files.append(os_path)
            if os_key in st.session_state.hearted:
                hearted_files.append(os_path)

            loop_os = np.tile(window, 4) if len(window) > 0 else np.zeros(int(sr * 0.5))
            loop_os_path = os.path.join(out_dir, f"osloop_{i}.wav")
            try:
                sf.write(loop_os_path, loop_os, sr)
            except Exception:
                sf.write(loop_os_path, np.zeros(int(sr * 0.5)), sr)
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
