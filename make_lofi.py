"""Agent Office FM - 오리지널 로파이 앰비언트 트랙 생성 (절차적 합성, 저작권フリー).
Dm9 - G13 - Cmaj9 - Am9 진행, 72 BPM, 펠트 피아노 + 웜 패드 + 브러시 드럼 + 바이닐 노이즈.
한 사이클(8마디)을 9번 반복하되 멜로디는 사이클마다 살짝 변주. 총 약 4분, 루프 재생용.
"""
import numpy as np, subprocess, os, sys

SR = 44100
BPM = float(sys.argv[1]) if len(sys.argv) > 1 else 72.0
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 20261029
PROG = int(sys.argv[3]) if len(sys.argv) > 3 else 0
OUTNAME = sys.argv[4] if len(sys.argv) > 4 else 'track1_lofi_office.mp3'
BEAT = 60.0 / BPM
BAR = BEAT * 4
rng_master = np.random.default_rng(SEED)

def mtof(m): return 440.0 * 2 ** ((m - 69) / 12.0)

# chord: (bass_midi, piano_voicing_midis, melody_pool_midis)
PROGS = [
    [   # 0: Dm9 - G13 - Cmaj9 - Am9
        (38, [53, 57, 60, 64], [62, 64, 65, 69, 72, 74]),
        (43, [53, 59, 64, 69], [62, 64, 67, 69, 71, 74]),
        (48, [55, 59, 62, 64], [60, 62, 64, 67, 71, 72]),
        (45, [52, 55, 59, 60], [60, 62, 64, 67, 69, 72]),
    ],
    [   # 1: Fmaj9 - Em7 - Dm9 - Cmaj9
        (41, [53, 57, 60, 64], [60, 62, 64, 65, 69, 72]),
        (40, [55, 59, 62, 65], [59, 62, 64, 67, 71, 72]),
        (38, [53, 57, 60, 64], [62, 64, 65, 69, 72, 74]),
        (48, [55, 59, 62, 64], [60, 62, 64, 67, 71, 72]),
    ],
    [   # 2: Am9 - Fmaj9 - Cmaj9 - G6
        (45, [52, 55, 59, 60], [60, 62, 64, 67, 69, 72]),
        (41, [53, 57, 60, 64], [60, 62, 64, 65, 69, 72]),
        (48, [55, 59, 62, 64], [60, 62, 64, 67, 71, 72]),
        (43, [52, 55, 59, 62], [59, 62, 64, 67, 71, 74]),
    ],
]
CHORDS = PROGS[PROG]
BARS_PER_CHORD = 2
CYCLE_BARS = len(CHORDS) * BARS_PER_CHORD      # 8 bars
CYCLE_SEC = CYCLE_BARS * BAR
CYCLES = 9
TOTAL_SEC = CYCLE_SEC * CYCLES
N = int(TOTAL_SEC * SR)
mix = np.zeros((N, 2), dtype=np.float32)

def add(start, sig, gain=1.0, pan=0.0):
    """stereo add with constant-power pan (-1..1)"""
    i0 = int(start * SR); i1 = min(N, i0 + len(sig))
    if i0 >= N: return
    s = sig[: i1 - i0] * gain
    l = np.cos((pan + 1) * np.pi / 4); r = np.sin((pan + 1) * np.pi / 4)
    mix[i0:i1, 0] += s * l; mix[i0:i1, 1] += s * r

def felt_piano(midi, dur=2.2, vel=1.0):
    f = mtof(midi); t = np.arange(int(dur * SR)) / SR
    harm = [(1.0, 1.0), (2.0, 0.45), (3.0, 0.20), (4.0, 0.10), (5.0, 0.045)]
    y = np.zeros_like(t)
    for ratio, amp in harm:
        det = 1.0 + (rng_master.random() - 0.5) * 0.0012
        y += amp * np.sin(2 * np.pi * f * ratio * det * t + rng_master.random() * 0.2)
    env = (1 - np.exp(-t / 0.006)) * np.exp(-t / (0.9 + vel * 0.7))
    # gentle lowpass (felt) via moving average
    k = 24; y = np.convolve(y, np.ones(k) / k, mode='same')
    return (y * env * vel).astype(np.float32)

def pad_tone(midi, dur, vel=1.0):
    f = mtof(midi); t = np.arange(int(dur * SR)) / SR
    y = np.zeros_like(t)
    for det_cents in (-5, 0, 5):
        d = 2 ** (det_cents / 1200.0)
        y += np.sin(2 * np.pi * f * d * t) + 0.4 * np.sin(2 * np.pi * f * 2 * d * t)
    env = np.minimum(1, t / 1.1) * np.minimum(1, (dur - t) / 1.4)
    env = np.clip(env, 0, 1) ** 1.5
    k = 220; y = np.convolve(y, np.ones(k) / k, mode='same')
    return (y * env * vel * 0.33).astype(np.float32)

def bass_note(midi, dur=1.6, vel=1.0):
    f = mtof(midi); t = np.arange(int(dur * SR)) / SR
    y = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * f * 2 * t) + 0.1 * np.sin(2 * np.pi * f * 3 * t)
    env = (1 - np.exp(-t / 0.01)) * np.exp(-t / 1.1)
    k = 60; y = np.convolve(y, np.ones(k) / k, mode='same')
    return (y * env * vel).astype(np.float32)

def kick(vel=1.0):
    dur = 0.24; t = np.arange(int(dur * SR)) / SR
    f = 120 * np.exp(-t / 0.05) + 46
    ph = np.cumsum(f) / SR * 2 * np.pi
    return (np.sin(ph) * np.exp(-t / 0.09) * vel).astype(np.float32)

def noise_burst(dur, hp=False, vel=1.0, swell=False):
    n = int(dur * SR); w = rng_master.standard_normal(n).astype(np.float32)
    if hp: w = np.diff(w, prepend=0) * 1.6
    else: w = np.convolve(w, np.ones(30) / 30, mode='same') * 2.2
    t = np.arange(n) / SR
    env = np.exp(-t / (dur * 0.55)) if not swell else (np.minimum(1, t / (dur * 0.6)) * np.exp(-t / (dur * 0.9)))
    return (w * env * vel).astype(np.float32)

for cyc in range(CYCLES):
    rng = np.random.default_rng(SEED % 100000 + cyc)
    base = cyc * CYCLE_SEC
    for ci, (bass_m, voice, pool) in enumerate(CHORDS):
        cs = base + ci * BARS_PER_CHORD * BAR
        # pads: whole chord span
        for j, m in enumerate(voice):
            add(cs, pad_tone(m, BARS_PER_CHORD * BAR, 0.5), gain=0.16, pan=-0.25 + 0.17 * j)
        # piano chords: strummed on bar starts + soft mid-bar
        for b in range(BARS_PER_CHORD):
            bs = cs + b * BAR
            for j, m in enumerate(voice):
                add(bs + j * 0.045 + rng.random() * 0.012, felt_piano(m, 2.4, 0.5), gain=0.20, pan=-0.15 + 0.1 * j)
            if rng.random() < 0.75:
                for j, m in enumerate(voice[:3]):
                    add(bs + BAR / 2 + j * 0.05, felt_piano(m, 1.8, 0.32), gain=0.16, pan=-0.1 + 0.1 * j)
        # bass: beats 1, 3 (+ passing on 4 sometimes)
        for b in range(BARS_PER_CHORD):
            bs = cs + b * BAR
            add(bs, bass_note(bass_m, 1.7, 0.9), gain=0.5)
            add(bs + 2 * BEAT, bass_note(bass_m, 1.4, 0.7), gain=0.45)
            if rng.random() < 0.4:
                add(bs + 3.5 * BEAT, bass_note(bass_m + 7, 0.7, 0.5), gain=0.4)
        # melody: swung eighths, sparse
        steps = BARS_PER_CHORD * 8
        last = None
        for s_i in range(steps):
            if rng.random() < 0.62: continue
            swing = 0.06 * BEAT if s_i % 2 else 0.0
            t0 = cs + s_i * (BAR / 8) + swing
            cand = [m for m in pool if last is None or abs(m - last) <= 7]
            m = int(rng.choice(cand)); last = m
            add(t0, felt_piano(m, 2.6, 0.62), gain=0.17, pan=0.18)
    # drums per bar
    for b in range(CYCLE_BARS):
        bs = base + b * BAR
        add(bs, kick(0.8), gain=0.5)
        add(bs + 2 * BEAT, kick(0.65), gain=0.45)
        add(bs + BEAT, noise_burst(0.28, hp=False, vel=0.5, swell=True), gain=0.10)       # brush snare
        add(bs + 3 * BEAT, noise_burst(0.28, hp=False, vel=0.45, swell=True), gain=0.10)
        for s_i in range(8):  # hats with swing
            swing = 0.05 * BEAT if s_i % 2 else 0.0
            add(bs + s_i * (BAR / 8) + swing, noise_burst(0.045, hp=True, vel=0.30), gain=0.055, pan=0.1)

# 노이즈 제거 (2026-10-29 사용자 지시: 바이닐 크래클·룸톤 없이 음악만 깨끗하게)

# dotted-eighth piano echo (cheap feedback delay on whole mix, subtle)
d = int(BEAT * 0.75 * SR)
echo = np.zeros_like(mix); echo[d:] += mix[:-d] * 0.16; echo[2 * d:] += mix[:-2 * d] * 0.07
mix += echo

# soft saturation + normalize + fades
mix = np.tanh(mix * 1.15)
peak = np.max(np.abs(mix)); mix = mix / max(peak, 1e-6) * 0.86
fade = int(1.2 * SR)
mix[:fade] *= np.linspace(0, 1, fade)[:, None]
mix[-fade:] *= np.linspace(1, 0, fade)[:, None]

out_wav = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'music', '_lofi_raw.wav')
os.makedirs(os.path.dirname(out_wav), exist_ok=True)
import wave
with wave.open(out_wav, 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(mix, -1, 1) * 32767).astype(np.int16).tobytes())
out_mp3 = out_wav.replace('_lofi_raw.wav', OUTNAME)
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', out_wav, '-codec:a', 'libmp3lame', '-b:a', '192k', out_mp3], check=True)
os.remove(out_wav)
print('OK', out_mp3, round(os.path.getsize(out_mp3) / 1e6, 2), 'MB, total', round(TOTAL_SEC, 1), 'sec')
