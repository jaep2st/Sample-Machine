import streamlit as st
import os
import tempfile
import zipfile
import json
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sample Machine", page_icon="🎧", layout="centered")
st.title("Sample Machine")
st.caption("Best chops · mute · make all loops · sample map")

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

def norm(seg, gain=1.0):
    peak =
