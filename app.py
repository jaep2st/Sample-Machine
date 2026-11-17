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
st.caption("Best chops · keep original · groups · before/after")

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

def apply_swing(seg, sr, swing_amount):
    if swing_amount <= 0.05 or len(seg) < sr * 0.3:
        return seg
    third = len(seg) // 3
    delay = int(swing_amount * 0.03 * sr)
    delay = min(delay, third // 2)
    if delay <= 0 or third <= delay:
        return seg
    swung = np.zeros_like(seg)
    end1 = min(third, len(seg))
    swung[:end1] = seg[:end1]
    start2 = third + delay
    end2 = min(start2 + third, len(seg))
    src_start = third
    src_end = min(src_start + (end2 - start2), len(seg))
    if start2 < len(swung) and src_end > src_start:
        swung[start2:end2] = seg[src_start:src_end]
    return swung

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

def find_best_chops(y, sr, bars=2, max_chops=6, weirder=False):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)
    song_len = len(y) / sr
    duration = (bars * 4 * 60) / bpm
    
    candidates = []
    for start in onsets:
        end = start + duration
        if end > song_len - 0.1:
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
        if weirder:
            score = score * (0.4 + np.random.uniform(0, 1.2)) * (1.0 - start / song_len * 0.4)
        candidates.append((start, end, score))
    
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    selected = []
    for s, e, sc in candidates:
        if all(not (s < se and e > ss) for ss, se, _ in selected):
            selected.append((s, e, sc))
        if len(selected) >= max_chops:
            break
    
    if selected and all(s > song_len * 0.5 for s, e, sc in selected):
        first_half = [c for c in candidates if c[0] < song_len * 0.5]
        if first_half:
            first_half.sort(key=lambda x: x[2], reverse=True)
            selected[-1] = first_half[0]
    
    return selected, bpm, beat_times

def extract_oneshots(y, sr, max_shots=8):
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True, delta=0.05)
    shots = []
    for onset in onsets[:150]:
        window = y[int(onset*sr):int((onset+0.18)*sr)]
        if len(window) < 40:
            continue
        # Favor the hit (first 80ms) over the tail
        hit = window[:int(0.08*sr)]
        tail = window[int(0.08*sr):]
        hit_energy = float(np.mean(np.abs(hit))) if len(hit) else 0
        tail_energy = float(np.mean(np.abs(tail))) if len(tail) else 0
        peak = float(np.max(np.abs(window)))
        if hit_energy > 0.08 and peak > 0.25 and hit_energy > tail_energy * 1.3:
            shots.append((onset, hit_energy * peak))
    shots.sort(key=lambda x: x[1], reverse=True)
    return shots[:max_shots]

def snap_to_nearest_beat(time, beat_times):
    if len(beat_times) == 0:
        return time
    return beat_times[np.argmin(np.abs(beat_times - time))]

def set_chop_bars(chop, bars, bpm, duration):
    target_len = (bars * 4 * 60) / bpm
    chop["end"] = min(duration, chop["start"] + target_len)

uploaded = st.file_uploader("Drop a track", type=["mp3", "wav", "m4a", "flac"])
session_upload = st.file_uploader("Load previous session (JSON)", type=["json"])

if session_upload:
    try:
        session_data = json.load(session_upload)
        st.session_state.auto_chops = session_data.get("chops", [])
        st.session_state.bpm = session_data.get("bpm", 90)
        st.session_state.key = session_data.get("key", "C")
        st.success("Session loaded")
    except Exception:
        st.error("Could not load session")

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

    lock_bpm = st.checkbox("Lock chops to target BPM", value=False)
    target_bpm = st.number_input("Target BPM", 60, 200, 90) if lock_bpm else None
    weirder = st.checkbox("Weirder chops", value=False)

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

    source_key = f"{uploaded.name}_{region_offset}_{target_bpm}_{weirder}"
    if "auto_chops" not in st.session_state or st.session_state.get("source_key") != source_key:
        with st.spinner("Finding the best chops..."):
            selected, detected_bpm, beat_times = find_best_chops(y_source, sr, bars=2, max_chops=6, weirder=weirder)
            bpm = target_bpm if lock_bpm else detected_bpm
            st.session_state.auto_chops = []
            st.session_state.bpm = bpm
            st.session_state.key = key
            st.session_state.beat_times = beat_times + region_offset
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
                    "eq_amount": 0.7,
                    "fade_in": 6,
                    "fade_out": 6,
                    "crossfade": 12,
                    "volume": 1.0,
                    "include": True,
                    "humanize": False,
                    "locked": False,
                    "swing": 0.0,
                    "solo": False,
                    "mute": False,
                    "favorite": False,
                    "keep_original": False,
                    "group": ""
                })

    bpm = st.session_state.bpm
    beat_times = st.session_state.get("beat_times", np.array([]))
    st.write(f"**{bpm:.1f} BPM** · **{key}**")

    if st.button("Copy last chop settings to all"):
        if st.session_state.auto_chops:
            src = st.session_state.auto_chops[-1]
            for chop in st.session_state.auto_chops:
                if not chop["locked"]:
                    chop["pitch"] = src["pitch"]
                    chop["reverse"] = src["reverse"]
                    chop["style"] = src["style"]
                    chop["eq"] = src["eq"]
                    chop["eq_amount"] = src["eq_amount"]
                    chop["fade_in"] = src["fade_in"]
                    chop["fade_out"] = src["fade_out"]
                    chop["crossfade"] = src["crossfade"]
                    chop["volume"] = src["volume"]
                    chop["swing"] = src["swing"]
            st.rerun()

    col_aud, col_fav = st.columns(2)
    with col_aud:
        if st.button("Audition all"):
            st.info("Playing all chops")
    with col_fav:
        favs = [c for c in st.session_state.auto_chops if c["favorite"]]
        if favs and st.button("Play favorites"):
            st.info(f"Playing {len(favs)} favorites")

    col_dup, col_all = st.columns(2)
    with col_dup:
        if st.button("Duplicate last chop"):
            if st.session_state.auto_chops:
                last = st.session_state.auto_chops[-1].copy()
                last["id"] = max(c["id"] for c in st.session_state.auto_chops) + 1
                last["locked"] = False
                last["solo"] = False
                last["mute"] = False
                last["favorite"] = False
                last["keep_original"] = False
                st.session_state.auto_chops.append(last)
                st.rerun()
    with col_all:
        if st.button("Make all loops"):
            for chop in st.session_state.auto_chops:
                if not chop["locked"]:
                    chop["start"] = snap_to_nearest_beat(chop["start"], beat_times)
                    chop["end"] = snap_to_nearest_beat(chop["end"], beat_times)
                    chop["crossfade"] = max(chop["crossfade"], 18)
                    chop["fade_in"] = max(chop["fade_in"], 8)
                    chop["fade_out"] = max(chop["fade_out"], 8)
            st.rerun()

    if st.button("Duplicate last + pitch variants"):
        if st.session_state.auto_chops:
            base = st.session_state.auto_chops[-1].copy()
            for p in [-5, -3, 3, 5]:
                new = base.copy()
                new["id"] = max(c["id"] for c in st.session_state.auto_chops) + 1
                new["pitch"] = p
                new["locked"] = False
                new["favorite"] = False
                new["keep_original"] = False
                st.session_state.auto_chops.append(new)
            st.rerun()

    if "presets" not in st.session_state:
        st.session_state.presets = {}
    preset_name = st.text_input("Preset name")
    if st.button("Save current settings as preset") and preset_name:
        if st.session_state.auto_chops:
            st.session_state.presets[preset_name] = {
                "pitch": st.session_state.auto_chops[0]["pitch"],
                "style": st.session_state.auto_chops[0]["style"],
                "eq": st.session_state.auto_chops[0]["eq"],
                "eq_amount": st.session_state.auto_chops[0]["eq_amount"],
                "fade_in": st.session_state.auto_chops[0]["fade_in"],
                "fade_out": st.session_state.auto_chops[0]["fade_out"],
                "crossfade": st.session_state.auto_chops[0]["crossfade"],
                "volume": st.session_state.auto_chops[0]["volume"],
                "swing": st.session_state.auto_chops[0]["swing"]
            }
            st.success(f"Saved preset: {preset_name}")

    if st.session_state.presets:
        load_preset = st.selectbox("Load preset", [""] + list(st.session_state.presets.keys()))
        if load_preset and st.button("Apply preset"):
            p = st.session_state.presets[load_preset]
            for chop in st.session_state.auto_chops:
                if not chop["locked"]:
                    chop.update(p)
            st.success(f"Applied preset: {load_preset}")

    out_dir = tempfile.mkdtemp()
    included_files = []
    chops_only_files = []
    fav_files = []
    oneshot_files = []
    loop_points = []
    sample_map = []
    favorites = []
    reel_segments = []
    before_after_files = []

    for chop in st.session_state.auto_chops:
        with st.expander(f"Chop {chop['id']} · {chop['start']:.1f}s–{chop['end']:.1f}s", expanded=True):
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            with c1:
                chop["include"] = st.checkbox("Include", value=chop["include"], key=f"inc_{chop['id']}")
            with c2:
                chop["locked"] = st.checkbox("Lock", value=chop["locked"], key=f"lock_{chop['id']}")
            with c3:
                chop["solo"] = st.checkbox("Solo", value=chop["solo"], key=f"solo_{chop['id']}")
            with c4:
                chop["mute"] = st.checkbox("Mute", value=chop["mute"], key=f"mute_{chop['id']}")
            with c5:
                chop["favorite"] = st.checkbox("Favorite", value=chop["favorite"], key=f"fav_{chop['id']}")
            with c6:
                chop["keep_original"] = st.checkbox("Keep original", value=chop["keep_original"], key=f"keep_{chop['id']}")

            chop["group"] = st.text_input("Group", value=chop["group"], key=f"group_{chop['id']}", placeholder="verse / hook / etc")

            if st.button("Reset", key=f"reset_{chop['id']}"):
                chop["start"] = chop["orig_start"]
                chop["end"] = chop["orig_end"]
                chop["pitch"] = 0
                chop["reverse"] = False
                chop["style"] = "Clean"
                chop["eq"] = "None"
                chop["eq_amount"] = 0.7
                chop["fade_in"] = 6
                chop["fade_out"] = 6
                chop["crossfade"] = 12
                chop["volume"] = 1.0
                chop["swing"] = 0.0
                chop["humanize"] = False
                chop["keep_original"] = False
                st.rerun()

            if not chop["locked"]:
                st.markdown("**Length**")
                l1, l2, l3 = st.columns(3)
                if l1.button("1 bar", key=f"bar1_{chop['id']}"):
                    set_chop_bars(chop, 1, bpm, duration)
                    st.rerun()
                if l2.button("2 bar", key=f"bar2_{chop['id']}"):
                    set_chop_bars(chop, 2, bpm, duration)
                    st.rerun()
                if l3.button("4 bar", key=f"bar4_{chop['id']}"):
                    set_chop_bars(chop, 4, bpm, duration)
                    st.rerun()

                st.markdown("**Nudge**")
                n1, n2, n3, n4 = st.columns(4)
                if n1.button("Start -50ms", key=f"ns-{chop['id']}"):
                    chop["start"] = max(0, chop["start"] - 0.05)
                    st.rerun()
                if n2.button("Start +50ms", key=f"ns+{chop['id']}"):
                    chop["start"] = min(chop["end"] - 0.2, chop["start"] + 0.05)
                    st.rerun()
                if n3.button("End -50ms", key=f"ne-{chop['id']}"):
                    chop["end"] = max(chop["start"] + 0.2, chop["end"] - 0.05)
                    st.rerun()
                if n4.button("End +50ms", key=f"ne+{chop['id']}"):
                    chop["end"] = min(duration, chop["end"] + 0.05)
                    st.rerun()

                col_start, col_end = st.columns(2)
                with col_start:
                    start_trim = st.slider("Trim start", -0.4, 0.4, chop["start"] - chop["orig_start"], 0.01, key=f"st_{chop['id']}")
                with col_end:
                    end_trim = st.slider("Trim end", -0.4, 0.4, chop["end"] - chop["orig_end"], 0.01, key=f"et_{chop['id']}")

                chop["start"] = max(0, chop["orig_start"] + start_trim)
                chop["end"] = min(duration, chop["orig_end"] + end_trim)
                if chop["end"] <= chop["start"] + 0.15:
                    chop["end"] = chop["start"] + 0.15

                if st.button("Clean loop", key=f"clean_{chop['id']}"):
                    chop["start"] = snap_to_nearest_beat(chop["start"], beat_times)
                    chop["end"] = snap_to_nearest_beat(chop["end"], beat_times)
                    chop["crossfade"] = max(chop["crossfade"], 18)
                    chop["fade_in"] = max(chop["fade_in"], 8)
                    chop["fade_out"] = max(chop["fade_out"], 8)
                    st.rerun()

                chop["humanize"] = st.checkbox("Humanize start", value=chop["humanize"], key=f"hum_{chop['id']}")

                st.markdown("**Quick pitch**")
                qp1, qp2, qp3, qp4, qp5, qp6 = st.columns(6)
                if qp1.button("-7", key=f"p-7_{chop['id']}"):
                    chop["pitch"] = -7
                    st.rerun()
                if qp2.button("-5", key=f"p-5_{chop['id']}"):
                    chop["pitch"] = -5
                    st.rerun()
                if qp3.button("-3", key=f"p-3_{chop['id']}"):
                    chop["pitch"] = -3
                    st.rerun()
                if qp4.button("+3", key=f"p+3_{chop['id']}"):
                    chop["pitch"] = 3
                    st.rerun()
                if qp5.button("+5", key=f"p+5_{chop['id']}"):
                    chop["pitch"] = 5
                    st.rerun()
                if qp6.button("+7", key=f"p+7_{chop['id']}"):
                    chop["pitch"] = 7
                    st.rerun()

                c1, c2, c3 = st.columns(3)
                with c1:
                    chop["pitch"] = st.slider("Pitch", -12, 12, chop["pitch"], key=f"pitch_{chop['id']}")
                with c2:
                    rev = st.checkbox("Reverse", value=chop["reverse"], key=f"rev_{chop['id']}")
                    if rev != chop["reverse"]:
                        chop["reverse"] = rev
                        st.rerun()
                with c3:
                    chop["style"] = st.selectbox("Style", ["Clean", "Dusty", "Warm", "Dark"],
                                                 index=["Clean","Dusty","Warm","Dark"].index(chop["style"]),
                                                 key=f"style_{chop['id']}")

                c4, c5 = st.columns(2)
                with c4:
                    chop["eq"] = st.selectbox("EQ / Filter", ["None", "Low shelf", "High shelf", "Low-pass", "High-pass"],
                                              index=["None","Low shelf","High shelf","Low-pass","High-pass"].index(chop["eq"]),
                                              key=f"eq_{chop['id']}")
                with c5:
                    if chop["eq"] != "None":
                        chop["eq_amount"] = st.slider("Amount", 0.0, 1.0, chop["eq_amount"], 0.05, key=f"eqamt_{chop['id']}")

                c6, c7 = st.columns(2)
                with c6:
                    chop["fade_in"] = st.slider("Fade in", 0, 60, chop["fade_in"], 2, key=f"fi_{chop['id']}")
                with c7:
                    chop["fade_out"] = st.slider("Fade out", 0, 60, chop["fade_out"], 2, key=f"fo_{chop['id']}")

                chop["volume"] = st.slider("Volume", 0.6, 1.4, chop["volume"], 0.05, key=f"vol_{chop['id']}")
                chop["crossfade"] = st.slider("Loop crossfade", 0, 60, chop["crossfade"], 2, key=f"cf_{chop['id']}")
                chop["swing"] = st.slider("Swing", 0.0, 1.0, chop["swing"], 0.05, key=f"swing_{chop['id']}")

            # Context
            ctx_start = max(0, chop["start"] - 2.0)
            ctx_end = min(duration, chop["end"] + 2.0)
            ctx_seg = y_full[int(ctx_start*sr):int(ctx_end*sr)].copy()
            ctx_path = os.path.join(out_dir, f"context_{chop['id']}.wav")
            sf.write(ctx_path, ctx_seg, sr)
            st.markdown("**In context**")
            st.audio(ctx_path)

            # Solo in context
            solo_ctx = ctx_seg.copy()
            chop_offset = int((chop["start"] - ctx_start) * sr)
            chop_len = int((chop["end"] - chop["start"]) * sr)
            solo_ctx[chop_offset:chop_offset+chop_len] *= 0.05
            solo_path = os.path.join(out_dir, f"solo_ctx_{chop['id']}.wav")
            sf.write(solo_path, solo_ctx, sr)
            st.markdown("**Solo in context**")
            st.audio(solo_path)

            start_use = chop["start"]
            if chop["humanize"] and not chop["locked"]:
                start_use += np.random.uniform(-0.02, 0.02)
                start_use = max(0, start_use)

            start_sample = int(start_use * sr)
            end_sample = int(min(chop["end"], duration) * sr)
            raw_seg = y_full[start_sample:end_sample].copy()
            seg = raw_seg.copy()

            if chop["pitch"] != 0:
                seg = librosa.effects.pitch_shift(seg, sr=sr, n_steps=chop["pitch"])
            if chop["reverse"]:
                seg = do_reverse(seg, sr)
            seg = process(seg, chop["style"])
            seg = apply_eq(seg, sr, chop["eq"], chop["eq_amount"])
            seg = apply_fade(seg, chop["fade_in"], chop["fade_out"], sr)
            seg = apply_swing(seg, sr, chop["swing"])
            seg = norm(seg, chop["volume"])

            # Before/After file
            ba = np.concatenate([raw_seg, seg])
            ba_name = f"before_after_chop{chop['id']:02d}.wav"
            ba_path = os.path.join(out_dir, ba_name)
            sf.write(ba_path, ba, sr)
            before_after_files.append(ba_path)

            # A/B
            raw_path = os.path.join(out_dir, f"raw_{chop['id']}.wav")
            sf.write(raw_path, raw_seg, sr)
            st.markdown("**A/B**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.caption("Original")
                st.audio(raw_path)
            with col_b:
                st.caption("Processed")
                st.audio(os.path.join(out_dir, name) if False else raw_path)

            name = f"{int(bpm)}bpm_{key}_chop{chop['id']:02d}"
            if chop["pitch"] != 0:
                name += f"_p{chop['pitch']:+d}"
            if chop["reverse"]:
                name += "_rev"
            if chop["style"] != "Clean":
                name += f"_{chop['style'].lower()}"
            if chop["eq"] != "None":
                name += f"_{chop['eq'].lower().replace(' ','')}"
            if chop["favorite"]:
                name += "_fav"
            if chop["group"]:
                name += f"_{chop['group'].replace(' ','_')}"
            name += ".wav"

            tmp_path = os.path.join(out_dir, name)
            sf.write(tmp_path, seg, sr)
            should_include = chop["include"] and not (any(c["solo"] for c in st.session_state.auto_chops) and not chop["solo"]) and not chop["mute"]
            if should_include:
                included_files.append(tmp_path)
                chops_only_files.append(tmp_path)
                reel_segments.append(seg)
                loop_points.append(f"Chop {chop['id']}: {start_use:.3f}s - {chop['end']:.3f}s")
                sample_map.append(f"{chop['id']},{start_use:.3f},{chop['end']-start_use:.3f},{name}")
                if chop["favorite"]:
                    favorites.append(f"Chop {chop['id']} (favorite)")
                    fav_files.append(tmp_path)
                if chop["keep_original"]:
                    raw_keep = os.path.join(out_dir, name.replace(".wav", "_original.wav"))
                    sf.write(raw_keep, raw_seg, sr)
                    included_files.append(raw_keep)
                    chops_only_files.append(raw_keep)

            st.markdown("**Single**")
            st.audio(tmp_path)
            with open(tmp_path, "rb") as f:
                st.download_button("Download", f, file_name=name, mime="audio/wav", key=f"dl_{chop['id']}")

            loop_seg = do_reverse(seg, sr) if chop["reverse"] else seg
            looped = apply_crossfade_loop(loop_seg, sr, chop["crossfade"])
            loop_name = name.replace(".wav", "_loop.wav")
            loop_path = os.path.join(out_dir, loop_name)
            sf.write(loop_path, looped, sr)
            if should_include:
                included_files.append(loop_path)
                if chop["favorite"]:
                    fav_files.append(loop_path)
            st.markdown("**Loop**")
            st.audio(loop_path)

    # Before/After pack
    if before_after_files:
        zpath = os.path.join(out_dir, "before_after.zip")
        with zipfile.ZipFile(zpath, "w") as zf:
            for fpath in before_after_files:
                zf.write(fpath, Path(fpath).name)
        with open(zpath, "rb") as f:
            st.download_button("Download Before/After", f, file_name="before_after.zip", key="ba")

    # Preview reel
    if reel_segments:
        reel = np.concatenate(reel_segments)
        reel_path = os.path.join(out_dir, "preview_reel.wav")
        sf.write(reel_path, reel, sr)
        st.markdown("**Preview Reel**")
        st.audio(reel_path)
        with open(reel_path, "rb") as f:
            st.download_button("Download Preview Reel", f, file_name="preview_reel.wav", key="reel")

    if chops_only_files:
        zpath = os.path.join(out_dir, "chops_only.zip")
        with zipfile.ZipFile(zpath, "w") as zf:
            for fpath in chops_only_files:
                zf.write(fpath, Path(fpath).name)
        with open(zpath, "rb") as f:
            st.download_button("Download Chops Only", f, file_name="chops_only.zip", key="chopsonly")

    if fav_files:
        zpath = os.path.join(out_dir, "favorites.zip")
        with zipfile.ZipFile(zpath, "w") as zf:
            for fpath in fav_files:
                zf.write(fpath, Path(fpath).name)
        with open(zpath, "rb") as f:
            st.download_button("Download Favorites Pack", f, file_name="favorites.zip", key="favpack")

    if favorites or loop_points:
        note = f"BPM: {bpm:.1f}\nKey: {key}\n\nFavorites:\n" + ("\n".join(favorites) if favorites else "None")
        note += "\n\nLoop Points:\n" + "\n".join(loop_points)
        note_path = os.path.join(out_dir, "session_note.txt")
        with open(note_path, "w") as f:
            f.write(note)
        with open(note_path, "rb") as f:
            st.download_button("Download Session Note", f, file_name="session_note.txt", key="note")

    if sample_map:
        map_text = "Chop,Start(sec),Length(sec),Filename\n" + "\n".join(sample_map)
        map_path = os.path.join(out_dir, "sample_map.csv")
        with open(map_path, "w") as f:
            f.write(map_text)
        with open(map_path, "rb") as f:
            st.download_button("Download Sample Map", f, file_name="sample_map.csv", key="map")

    if st.button("Export session"):
        session_data = {"bpm": bpm, "key": key, "chops": st.session_state.auto_chops}
        session_path = os.path.join(out_dir, "session.json")
        with open(session_path, "w") as f:
            json.dump(session_data, f, default=str)
        with open(session_path, "rb") as f:
            st.download_button("Download Session", f, file_name="session.json", key="session")

    if loop_points:
        points_text = "\n".join(loop_points)
        points_path = os.path.join(out_dir, "loop_points.txt")
        with open(points_path, "w") as f:
            f.write(points_text)
        with open(points_path, "rb") as f:
            st.download_button("Download Loop Points", f, file_name="loop_points.txt", key="points")

    st.divider()
    st.subheader("One-shots")
    max_oneshots = st.slider("One-shots to extract", 4, 16, 8)
    oneshot_len = st.slider("One-shot length (sec)", 0.3, 1.2, 0.5, 0.05)
    oneshot_pitch = st.slider("One-shot pitch", -12, 12, 0)
    reverse_oneshots = st.checkbox("Reverse one-shots", value=False)
    oneshots = extract_oneshots(y_full, sr, max_shots=max_oneshots)
    include_oneshots = st.checkbox("Include one-shots in pack", value=True)

    for i, (onset, energy) in enumerate(oneshots, 1):
        window = y_full[int(onset*sr):int((onset + oneshot_len)*sr)].copy()
        if oneshot_pitch != 0:
            window = librosa.effects.pitch_shift(window, sr=sr, n_steps=oneshot_pitch)
        if reverse_oneshots:
            window = do_reverse(window, sr)
        window = apply_fade(window, 4, 4, sr)
        window = norm(window)
        os_name = f"{int(bpm)}bpm_{key}_oneshot{i:02d}_{onset:.1f}s"
        if oneshot_pitch != 0:
            os_name += f"_p{oneshot_pitch:+d}"
        if reverse_oneshots:
            os_name += "_rev"
        os_name += ".wav"
        os_path = os.path.join(out_dir, os_name)
        sf.write(os_path, window, sr)
        if include_oneshots:
            oneshot_files.append(os_path)
        oneshot_loop = np.concatenate([window] * 4)
        loop_os_path = os.path.join(out_dir, f"oneshot_loop_{i}.wav")
        sf.write(loop_os_path, oneshot_loop, sr)
        st.markdown(f"**One-shot {i}** · {onset:.1f}s")
        st.audio(os_path)
        st.audio(loop_os_path)
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
