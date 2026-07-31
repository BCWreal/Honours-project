#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RM – One-Bar Probe Experiment 1b (v0.1n)
v0.1n — add jittered step_actual logging and staircase viz uses presented offset; otherwise same as prior version.
Note: 0.1l uses 45-degree spacing but excludes 12/6 o'clock (90/270).
=======================================

Design summary
--------------
- Task: Probe Judgment (left/right of the remembered vanish position).
- One Moving bar
- Bar y matches the trial's polar coordinate cy; bar moves horizontally and
  vanishes at (cx, cy).
- Main manipulation: motion smoothness via stimulus-onset asynchrony (SOA).

  Four conditions, with on-frames / off-frames at 144 Hz:
      smooth   : on=  7 ms (1 frame),  off=  0 ms (0 frames)  — continuous
      soa_100  : on= 50 ms (7 frames), off= 50 ms (7 frames)  — SOA ≈  97.2 ms
      soa_200  : on=100 ms (14 frames), off=100 ms (14 frames) — SOA ≈ 194.4 ms
      soa_300  : on=150 ms (22 frames), off=150 ms (22 frames) — SOA ≈ 305.6 ms

- Position step per discrete update = bar_speed_px_per_sec × actual_SOA, so
  the implied speed is consistent with continuous motion condition.

- Vanish position is anchored at (cx, cy); start position adjusts so an
  integer number of discrete positions fits between start and vanish.
  → effective d_init may differ from nominal d_init by up to one position-step.

- Unified d_init pool across all SOA conditions: (360, 540, 720, 900).
  Values spaced at 180 px to give four distinct effective d_init at every SOA,
  including soa_300 (pos_step ≈ 183.33 px at 144 Hz). At soa_300, the four
  values yield 3, 4, 5, and 6 visible flashes (effective d_init = 367, 550, 733, 917 px).

- Experimenter-only staircase visualization window on a second monitor
    (two tracks, updated after each valid trial; fixation breaks are skipped).

Counterbalancing (per pid_index)
--------------------------------
- block_order_idx = pid_index % 4 → balanced Latin square order of 4 SOA blocks:
    0: [0,1,3,2], 1: [1,2,0,3], 2: [2,3,1,0], 3: [3,0,2,1]
- color_idx       = COLOR_SEQUENCE[pid_index % 8]
    COLOR_SEQUENCE = [B, R, B, R, R, B, R, B] (B=blue, R=red)
Full crossing every 8 participants.

Eye tracking (optional)
-----------------------
- EYE_MODE ∈ {"off", "dummy", "live"}, set in the GUI.
- "off"   : no EyeLink code runs. For debugging or sessions without a tracker.
- "dummy" : pylink loaded in dummy mode. Calibration UI works, no real samples.
- "live"  : full pylink integration, real-time gaze monitoring during motion.
- Calibration once at session start.
- Drift correction at the start of each SOA block.
- Online fixation enforcement during motion ONLY in the last 200 ms before
    target offset, and during the first 100 ms of the retention interval.
- Fixation break = gaze invalid OR outside fixation window for at least
    FIX_BREAK_CONSEC_FRAMES consecutive frames during the monitored period.
- On break: motion is aborted and the trial is re-rolled (new polar position,
  new d_init, new motion direction); the staircase LEVEL is preserved so the
  staircase doesn't burn through trials due to fixation breaks.

Logging
-------
Single unified CSV (one row per trial attempt). The `trial_outcome` column
distinguishes "ok" from "fixation_break". Failed trials carry stimulus and
level metadata but no probe / response / staircase-update fields. Reversal
sidecar CSV is unified. A meta JSON sidecar dumps the session config.

Output files (under <script_dir>/Exp1b_Data/)
---------------------------------------------
    "RMprobe1b_<pid>_<date>_<time>_v0_1n.csv            (main log)
    "RMprobe1b_<pid>_<date>_<time>_v0_1n_reversals.csv  (reversal log; optional)
    "RMprobe1b_<pid>_<date>_<time>_v0_1n_meta.json      (session config)
    "RMprobe1b_<pid>_<date>_<time>_v0_1n.EDF            (EyeLink data; live/dummy)

Compatibility with EyeLink
--------------------------
EyeLinkCoreGraphicsPsychoPy.py must sit in the same folder as this script
(matching the working EyeLink_Fixation_Monitor.py setup). pylink must be
importable. If either is unavailable the script will warn and force EYE_MODE
to "off".
"""

import os
import sys
import csv
import math
import json
import time
import random
import platform
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from contextlib import nullcontext
from collections import OrderedDict

import numpy as np
from psychopy import visual, core, event, gui, monitors, logging as psychopy_logging

psychopy_logging.console.setLevel(psychopy_logging.CRITICAL)

# Optional EyeLink imports — guarded so the script can run without pylink.
HAS_PYLINK = True
try:
    import pylink
    from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
except Exception as _e:
    HAS_PYLINK = False
    pylink = None
    EyeLinkCoreGraphicsPsychoPy = None
    _PYLINK_IMPORT_ERROR = str(_e)
else:
    _PYLINK_IMPORT_ERROR = ""


# =========================
# Hardware / display params
# =========================

HARDWARE = {
    "resolution":    (1920, 1080),
    "fullscreen":    True,
    "refresh_hz":    144,
    "mac_safe_flip": platform.system() == "Darwin",
    "scr_width_cm":  59.7,    # 27-inch monitor
    "viewing_dist_cm": 57.0,
}

STIM = {
    "bar_width":              20,
    "bar_height":             100,
    "bar_speed":              4,           # px/frame; recomputed at runtime from speed_px_per_sec
    "bar_speed_px_per_sec":   600.0,
    "fg_color":               "white",     # color of the moving bar (probe color is set separately)
    "bg_color":               [-0.2, -0.2, -0.2],
}

TIMING = {
    "pre_fix_sec":   1.5,
    "iti_sec":       1.0,
    "retention_sec": 0.25,   # blank between motion offset and probe onset; identical across SOAs
}

RADIUS_PX        = 120
POLAR_ANGLES_DEG = [0, 45, 135, 180, 225, 315]
N_POLAR_ANGLES   = len(POLAR_ANGLES_DEG)

RESPONSE_KEYS = {
    "left":     "f",
    "right":    "k",
    "quit":     "escape",
    "continue": "space",
}


# =========================
# SOA condition table
# =========================
# Each row: (label, nominal_on_ms, nominal_off_ms).
# Frame counts are derived from the actual refresh rate at runtime.
# Order in this list defines the index mapping used by the balanced Latin square.

SOA_CONDITIONS: List[Tuple[str, float, float]] = [
    ("smooth",   7.0,   0.0),
    ("soa_100", 50.0,  50.0),
    ("soa_200", 100.0, 100.0),
    ("soa_300", 150.0, 150.0),
]

SOA_LABELS = [c[0] for c in SOA_CONDITIONS]

# Unified d_init pool across all SOA conditions (px). Values spaced
# at 180 px to give four distinct effective d_init at every SOA,
# including soa_300 (pos_step ≈ 183.33 px at 144 Hz). At soa_300,
# the four values yield 3, 4, 5, and 6 visible flashes
# (effective d_init = 367, 550, 733, 917 px).
D_INIT_POOLS: Dict[str, Tuple[float, ...]] = {
    "smooth":  (360, 540, 720, 900),
    "soa_100": (360, 540, 720, 900),
    "soa_200": (360, 540, 720, 900),
    "soa_300": (360, 540, 720, 900),
}


# =========================
# Staircase parameters
# =========================
# Same as Exp 1a probe task. Each SOA block runs an independent pair of
# interleaved staircase tracks.
# 0.1k staircase rule updates (larger starting step, updated tightening, lower max reversals).

STAIRCASE = {
    "seeds":                    {"A": +60.0, "B": -60.0},
    "steps":                    (45.0, 27.0, 17.0, 10.0),
    "tighten_after_reversals":  (4, 5, 6, 7),
    "bounds":                   (-150.0, 150.0),
    "max_reversals_per_track":  22,
    "max_trials_per_condition": 150,
}

DEMO = {
    "enabled":                  False,
    "max_trials_per_condition": 20,
    "max_reversals_per_track":  6,
    "steps":                    (45.0, 27.0, 17.0),
    "tighten_after_reversals":  (1, 2),
}

SANITY = {
    "enabled": False,
}


# =========================
# Eye tracker config
# =========================

EYE = {
    "mode":                       "live",   # "off" | "dummy" | "live"
    "host_ip":                    "100.1.1.1",
    "fixation_window_deg":        3.0,
    "fix_break_consec_frames":    7,        # ≈49 ms at 144 Hz
    "pre_offset_monitor_ms":       200.0,
    "post_offset_monitor_ms":      200.0,
    "edf_basename":               "rm1b",   # ≤8 chars, no extension
    "log_edf":                     False,
    "missing_sentinel":           -32768.0,
    "calibration_type":           "HV9",
}


# =========================
# Logging
# =========================
# Single unified header. One row per trial attempt (including fixation breaks).

MAIN_HEADERS = [
    # session identity
    "exp_date", "exp_time", "participant", "age",
    "demo_mode", "eye_mode",
    # block / counterbalance
    "block_index", "block_soa_label",
    "block_order_idx", "block_order_str",
    "color_idx", "probe_color",
    # SOA condition specifics
    "soa_label",
    "on_dur_ms_nominal", "off_dur_ms_nominal",
    "on_frames", "off_frames",
    "soa_ms_actual", "pos_step_px",
    "n_positions", "motion_total_frames", "motion_total_dur_ms",
    "d_init_nominal", "d_init_effective",
    # trial identity
    "task",
    "global_trial", "block_trial", "block_trial_attempt",
    "staircase_id", "track_id", "track_trial", "start_level_seed",
    "trial_outcome",                  # "ok" | "fixation_break"
    "fixation_break_count_for_trial", # how many breaks before this attempt's outcome
    # geometry
    "radius_px", "polar_angle_deg", "cx", "cy", "bar_y",
    "start_side", "motion_dir",
    "bar_start_x", "bar_vanish_x",
    # timing context
    "refresh_hz", "pre_fix_frames", "retention_frames", "iti_frames",
    "bar_speed_px_per_frame", "speed_px_per_sec",
    "retention_interval_ms",
    "n_frames_displayed",
    # staircase (level controls probe offset along motion direction)
    "level_before", "step_level", "step_size_px", "step_actual",
    "probe_offset_motion", "probe_x",
    "x_clean", "probe_jitter", "x_presented",
    # response
    "resp_left", "resp_right", "resp_key", "rt_ms",
    "resp_forward", "resp_backward",
    # staircase update
    "update_dir", "step_size_before", "step_size_after", "tightened_step",
    "level_after", "n_reversals", "rev_index", "is_reversal", "rev_level",
    "rev_count_level", "rev_count_total", "is_reversal_this_trial",
    "bounds_low", "bounds_high",
    # eye-tracking diagnostics for this attempt
    "fix_break_at_position", "fix_break_at_motion_frame",
    # modeling-friendly columns
    "x_motion", "y_forward",
]

REV_HEADERS = [
    "exp_date", "exp_time", "participant", "age", "demo_mode", "eye_mode",
    "block_index", "soa_label", "task", "staircase_id",
    "speed_px_per_sec", "retention_interval_ms",
    "track_trial_at_reversal", "rev_index", "rev_level",
    "update_dir_that_caused", "step_before", "step_after",
    "bounds_low", "bounds_high",
    "block_order_idx", "block_order_str", "color_idx", "probe_color",
]


# =========================
# Helpers — counterbalance
# =========================

def get_pid_index(participant: str) -> int:
    digits = "".join([c for c in participant if c.isdigit()])
    if digits:
        try:
            return int(digits)
        except Exception:
            pass
    return sum(ord(c) for c in participant)


# Balanced Latin square orders (indices into SOA_LABELS).
BALANCED_LATIN_SQUARE: List[Tuple[int, ...]] = [
    (0, 1, 3, 2),
    (1, 2, 0, 3),
    (2, 3, 1, 0),
    (3, 0, 2, 1),
]
BLOCK_ORDERS: List[Tuple[str, ...]] = [
    tuple(SOA_LABELS[i] for i in order) for order in BALANCED_LATIN_SQUARE
]
assert len(BLOCK_ORDERS) == 4

COLOR_SEQUENCE = [0, 1, 0, 1, 1, 0, 1, 0]  # 0=blue, 1=red


def get_counterbalance(pid_index: int) -> Dict[str, Any]:
    color_idx       = COLOR_SEQUENCE[pid_index % len(COLOR_SEQUENCE)]
    block_order_idx = pid_index % len(BLOCK_ORDERS)
    block_order     = list(BLOCK_ORDERS[block_order_idx])
    probe_color     = "blue" if color_idx == 0 else "red"
    return {
        "color_idx":       color_idx,
        "probe_color":     probe_color,
        "block_order_idx": block_order_idx,
        "block_order":     block_order,
        "block_order_str": "->".join(block_order),
    }


def get_staircase_params() -> Dict:
    sc = dict(STAIRCASE)
    if DEMO.get("enabled", False):
        sc["max_trials_per_condition"] = int(DEMO["max_trials_per_condition"])
        sc["max_reversals_per_track"]  = int(DEMO["max_reversals_per_track"])
        sc["steps"]                    = tuple(float(x) for x in DEMO["steps"])
        sc["tighten_after_reversals"]  = tuple(int(x) for x in DEMO["tighten_after_reversals"])
    return sc


# =========================
# Helpers — SOA / trajectory
# =========================

def compute_soa_frame_counts(refresh_hz: int) -> Dict[str, Dict[str, float]]:
    """For each SOA condition, compute on/off frame counts and derived quantities."""
    frame_period_ms = 1000.0 / float(refresh_hz)
    out = {}
    for label, on_ms, off_ms in SOA_CONDITIONS:
        on_frames  = max(1, int(round(on_ms  / frame_period_ms))) if on_ms  > 0 else 0
        off_frames = max(0, int(round(off_ms / frame_period_ms))) if off_ms > 0 else 0
        # Force smooth condition to be exactly 1/0 even if 7 ms doesn't round to 1.
        if label == "smooth":
            on_frames  = 1
            off_frames = 0
        soa_frames = on_frames + off_frames
        soa_ms_actual = soa_frames * frame_period_ms
        out[label] = {
            "on_dur_ms_nominal":  on_ms,
            "off_dur_ms_nominal": off_ms,
            "on_frames":          on_frames,
            "off_frames":         off_frames,
            "soa_frames":         soa_frames,
            "soa_ms_actual":      soa_ms_actual,
        }
    return out


def compute_trajectory(
    soa_info: Dict[str, float],
    d_init_nominal: float,
    bar_speed_px_per_frame: float,
) -> Dict[str, float]:
    """Compute trajectory geometry for one trial.

    Anchors the vanish position; effective d_init = (n_positions-1) * pos_step,
    so n_positions adjusts to make the discretization fit.
    """
    on_frames  = soa_info["on_frames"]
    off_frames = soa_info["off_frames"]
    soa_frames = on_frames + off_frames

    # Position step (px) per discrete update.
    # In the smooth condition (off=0), this is just bar_speed_px_per_frame.
    pos_step = bar_speed_px_per_frame * float(soa_frames)

    n_steps = max(1, int(round(d_init_nominal / pos_step))) if pos_step > 0 else 1
    n_positions = n_steps + 1
    d_init_effective = n_steps * pos_step

    # Total motion duration: N positions × on_frames + (N-1) gaps × off_frames.
    motion_total_frames = n_positions * on_frames + (n_positions - 1) * off_frames

    return {
        "pos_step":            pos_step,
        "n_steps":             n_steps,
        "n_positions":         n_positions,
        "d_init_effective":    d_init_effective,
        "motion_total_frames": motion_total_frames,
    }


def positions_for_trial(traj: Dict[str, float], cx: float, motion_dir: int) -> List[float]:
    """List of x-coords for the bar at each on-phase position.

    Vanish is the LAST element. motion_dir = +1 → motion is rightward
    (start on left of cx); motion_dir = -1 → motion is leftward.
    """
    n = traj["n_positions"]
    s = traj["pos_step"]
    return [cx - motion_dir * (n - 1 - i) * s for i in range(n)]


def enumerate_polar_positions() -> List[Tuple[float, float, float]]:
    angles_deg = list(POLAR_ANGLES_DEG)
    out = []
    for adeg in angles_deg:
        arad = math.radians(adeg)
        cx = RADIUS_PX * math.cos(arad)
        cy = RADIUS_PX * math.sin(arad)
        out.append((cx, cy, adeg))
    return out


# =========================
# Helpers — display
# =========================

def safe_flip(win):
    if HARDWARE.get("mac_safe_flip", False):
        try:
            win.flip()
        except AttributeError:
            try:
                win.flip()
            except AttributeError:
                pass
    else:
        win.flip()


def activate_window(win):
    try:
        handle = getattr(win, "winHandle", None)
        if handle is not None and hasattr(handle, "activate"):
            handle.activate()
    except Exception:
        pass


def make_stim_objects(win):
    fixation = visual.Circle(
        win, radius=6,
        fillColor=STIM["fg_color"], lineColor=STIM["fg_color"],
        edges=64, pos=(0, 0),
    )
    bar = visual.Rect(
        win, width=STIM["bar_width"], height=STIM["bar_height"],
        fillColor=STIM["fg_color"], lineColor=STIM["fg_color"],
        pos=(0, 0),
    )
    bar_ghost = visual.Rect(
        win, width=STIM["bar_width"], height=STIM["bar_height"],
        fillColor=None, lineColor="yellow",
        pos=(0, 0),
    )
    sanity_circle = visual.Circle(
        win, radius=RADIUS_PX,
        fillColor=None, lineColor="yellow",
        edges=128, pos=(0, 0),
    )
    return fixation, bar, bar_ghost, sanity_circle


def draw_accumulated_flashes(fixation, bar, sanity_circle, positions, bar_y):
    sanity_circle.draw()
    fixation.draw()
    for x in positions:
        bar.pos = (x, bar_y)
        bar.draw()


def enforce_fixation_for_frames(win, fixation, eyetracker, n_frames, fix_break_consec_threshold):
    """Enforce fixation for a fixed number of frames (post-offset window)."""
    consec_invalid = 0
    prev_sample = None
    last_in_window = True
    for frame_idx in range(n_frames):
        fixation.draw()
        safe_flip(win)

        if eyetracker.is_active():
            in_win, sample = eyetracker.get_gaze(prev_sample)
            if in_win is None:
                in_win = last_in_window
            else:
                last_in_window = in_win
            prev_sample = sample

            if not in_win:
                consec_invalid += 1
                if consec_invalid >= fix_break_consec_threshold:
                    return {"status": "fixation_break", "broke_at_frame": frame_idx}
            else:
                consec_invalid = 0

    return {"status": "ok", "broke_at_frame": -1}


def wait_for_keypress(win, allowed_keys, quit_key="escape", draw_func=None):
    allowed = [k.lower() for k in allowed_keys]
    quit_k  = quit_key.lower() if quit_key else None
    activate_window(win)
    event.clearEvents(eventType="keyboard")
    key_list = allowed + ([quit_k] if quit_k and quit_k not in allowed else [])
    while True:
        activate_window(win)
        if draw_func:
            draw_func()
        safe_flip(win)
        keys = event.getKeys(keyList=key_list)
        if not keys:
            keys = event.waitKeys(maxWait=0.05, keyList=key_list, clearEvents=False) or []
        if keys:
            k = keys[0].lower()
            if quit_k and k == quit_k:
                return "quit"
            if k in allowed:
                event.clearEvents(eventType="keyboard")
                return k


def show_text_screen(win, lines, key="space"):
    txt = visual.TextStim(
        win, text="\n".join(lines),
        pos=(0, 0), height=26, color="white",
        alignText="center", wrapWidth=1200,
    )
    return wait_for_keypress(win, [key], quit_key=RESPONSE_KEYS["quit"], draw_func=txt.draw)


def show_brief_message(win, text, duration_sec=1.0, color="yellow", refresh_hz=144):
    msg = visual.TextStim(win, text=text, pos=(0, 0), height=32, color=color, bold=True)
    n_frames = int(round(duration_sec * refresh_hz))
    for _ in range(n_frames):
        msg.draw()
        safe_flip(win)


def wait_for_lr_keypress(win, draw_func):
    event.clearEvents(eventType="keyboard")
    key_aliases = {
        RESPONSE_KEYS["left"].lower():  False,
        RESPONSE_KEYS["right"].lower(): True,
        "left": False, "right": True, "f": False, "k": True,
    }
    quit_key = RESPONSE_KEYS["quit"].lower()
    key_list = sorted(set(list(key_aliases.keys()) + [quit_key]))
    activate_window(win)
    draw_func()
    safe_flip(win)
    rt_clock = core.Clock()
    while True:
        activate_window(win)
        keys = event.getKeys(keyList=key_list)
        if keys:
            k = keys[0].lower()
            rt_ms = int(rt_clock.getTime() * 1000)
            if k == quit_key:
                return "quit", rt_ms
            if k in key_aliases:
                return key_aliases[k], rt_ms
        draw_func()
        safe_flip(win)


# =========================
# Helpers — staircase visualization
# =========================

def create_staircase_window():
    try:
        return visual.Window(
            size=[1600, 900],
            screen=1,
            fullscr=False,
            winType="pyglet",
            units="pix",
            color=[-0.9, -0.9, -0.9],
            allowGUI=False,
            waitBlanking=False,
            autoLog=False,
            checkTiming=False,
        )
    except Exception as e:
        print("Staircase window unavailable (continuing without it):", e)
        return None


def show_staircase_startup(win):
    if win is None:
        return
    activate_window(win)
    msg = visual.TextStim(
        win,
        text="Staircase window ready\nWaiting for trials...",
        pos=(0, 0),
        height=24,
        color="white",
        alignText="center",
    )
    msg.draw()
    win.flip()


class StaircaseVisualizer:
    def __init__(self, win):
        self.win = win
        self.bounds = (-150.0, 150.0)
        self.title = "Staircase Tracks"
        self.subtitle = ""
        self.data = {"A": [], "B": []}

    def reset(self, title: str, bounds: Tuple[float, float], subtitle: str = ""):
        if self.win is None:
            return
        self.title = title
        self.subtitle = subtitle
        self.bounds = bounds
        self.data = {"A": [], "B": []}
        self.draw()

    def add_point(self, track_id: str, level: float, subtitle: Optional[str] = None):
        if self.win is None:
            return
        if track_id not in self.data:
            return
        self.data[track_id].append(float(level))
        if subtitle is not None:
            self.subtitle = subtitle
        self.draw()

    def draw(self):
        if self.win is None:
            return

        w, h = self.win.size
        margin = 60
        x0 = -w / 2 + margin
        x1 = w / 2 - margin
        y0 = -h / 2 + margin
        y1 = h / 2 - margin

        low, high = self.bounds
        span = (high - low) if (high - low) != 0 else 1.0

        def _x(i: int, n: int) -> float:
            if n <= 1:
                return x0
            return x0 + (x1 - x0) * (i / float(n - 1))

        def _y(val: float) -> float:
            return y0 + (y1 - y0) * ((val - low) / span)

        plot_box = visual.Rect(
            self.win,
            width=(x1 - x0),
            height=(y1 - y0),
            pos=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
            lineColor="white",
            fillColor=None,
        )
        plot_box.draw()

        if low <= 0.0 <= high:
            y_zero = _y(0.0)
            visual.Line(self.win, start=(x0, y_zero), end=(x1, y_zero), lineColor="gray").draw()

        nA = len(self.data["A"])
        nB = len(self.data["B"])
        n_max = max(nA, nB, 1)

        def _draw_track(values: List[float], color: str):
            if not values:
                return
            last_pos = None
            for i, v in enumerate(values):
                pos = (_x(i, n_max), _y(v))
                if last_pos is not None:
                    visual.Line(self.win, start=last_pos, end=pos, lineColor=color).draw()
                visual.Circle(self.win, radius=4, pos=pos, fillColor=color, lineColor=color).draw()
                last_pos = pos

        _draw_track(self.data["A"], "dodgerblue")
        _draw_track(self.data["B"], "magenta")

        title = visual.TextStim(
            self.win, text=self.title, pos=(0, h / 2 - 24), height=24, color="white"
        )
        title.draw()

        subtitle = visual.TextStim(
            self.win, text=self.subtitle, pos=(0, h / 2 - 50), height=18, color="white"
        )
        subtitle.draw()

        a_last = self.data["A"][-1] if self.data["A"] else None
        b_last = self.data["B"][-1] if self.data["B"] else None
        a_txt = "A: --" if a_last is None else "A: {:.1f}".format(a_last)
        b_txt = "B: --" if b_last is None else "B: {:.1f}".format(b_last)
        status = "{} (n={})   {} (n={})".format(a_txt, nA, b_txt, nB)

        status_txt = visual.TextStim(
            self.win, text=status, pos=(0, -h / 2 + 24), height=18, color="white"
        )
        status_txt.draw()

        self.win.flip()


# =========================
# Helpers — EyeLink integration
# =========================

class EyeLinkSession:
    """Wrapper around pylink for the lifetime of one experimental session.

    Methods are no-ops if mode == 'off'.
    """

    def __init__(self, mode: str, edf_basename: str, host_ip: str,
                 calibration_type: str = "HV9", log_edf: bool = True):
        self.mode = mode
        self.edf_basename = edf_basename[:8]   # EyeLink filename limit
        self.edf_filename = self.edf_basename + ".EDF"
        self.host_ip = host_ip
        self.calibration_type = calibration_type
        self.log_edf = bool(log_edf)
        self.tracker = None
        self.genv = None
        self.scn_w = 0
        self.scn_h = 0
        self.fix_el_x = 0.0
        self.fix_el_y = 0.0
        self.fix_win_px = 0.0
        self.eye_used = 0
        self.eyelink_ver = 0

    def is_active(self) -> bool:
        return self.mode in ("dummy", "live") and self.tracker is not None

    def open(self):
        if self.mode == "off":
            return
        if not HAS_PYLINK:
            raise RuntimeError(
                "EyeLink mode '{}' requested but pylink could not be imported "
                "(error: {}). Install pylink or set EyeLink Mode = off."
                .format(self.mode, _PYLINK_IMPORT_ERROR)
            )
        if self.mode == "dummy":
            self.tracker = pylink.EyeLink(None)
        else:
            try:
                self.tracker = pylink.EyeLink(self.host_ip)
            except RuntimeError as e:
                raise RuntimeError("Failed to connect to EyeLink at {}: {}".format(self.host_ip, e))

        self.tracker.openDataFile(self.edf_filename)
        if self.log_edf:
            self.tracker.sendCommand("add_file_preamble_text 'RM Exp1b probe'")
        self.tracker.setOfflineMode()

        if self.mode == "live":
            vstr = self.tracker.getTrackerVersionString()
            try:
                self.eyelink_ver = int(vstr.split()[-1].split('.')[0])
            except Exception:
                self.eyelink_ver = 0

        link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'
        if self.eyelink_ver > 3:
            link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
        else:
            link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
        if self.log_edf:
            file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
            if self.eyelink_ver > 3:
                file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
            else:
                file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
            self.tracker.sendCommand("file_event_filter = " + file_event_flags)
            self.tracker.sendCommand("file_sample_data  = " + file_sample_flags)
        self.tracker.sendCommand("link_event_filter = " + link_event_flags)
        self.tracker.sendCommand("link_sample_data  = " + link_sample_flags)
        self.tracker.sendCommand("calibration_type = " + self.calibration_type)

    def setup_graphics(self, win, fix_window_deg: float):
        if not self.is_active():
            return
        self.scn_w, self.scn_h = win.size
        self.tracker.sendCommand(
            "screen_pixel_coords = 0 0 {:d} {:d}".format(self.scn_w - 1, self.scn_h - 1)
        )
        self.tracker.sendMessage(
            "DISPLAY_COORDS 0 0 {:d} {:d}".format(self.scn_w - 1, self.scn_h - 1)
        )

        self.genv = EyeLinkCoreGraphicsPsychoPy(self.tracker, win)
        self.genv.setCalibrationColors((-1, -1, -1), win.color)
        self.genv.setTargetType("circle")
        self.genv.setTargetSize(24)
        self.genv.setCalibrationSounds("", "", "")
        pylink.openGraphicsEx(self.genv)

        # Fixation window (px). Note EyeLink coords are top-left origin.
        px_per_cm  = self.scn_w / float(HARDWARE["scr_width_cm"])
        cm_per_deg = HARDWARE["viewing_dist_cm"] * math.tan(math.radians(1.0))
        self.fix_win_px = fix_window_deg * (px_per_cm * cm_per_deg)
        self.fix_el_x = self.scn_w / 2.0
        self.fix_el_y = self.scn_h / 2.0

    def calibrate(self):
        if not self.is_active():
            return
        try:
            self.tracker.doTrackerSetup()
        except RuntimeError as e:
            print("EyeLink calibration error:", e)
            try:
                self.tracker.exitCalibration()
            except Exception:
                pass

    def drift_correct(self):
        if not self.is_active():
            return
        try:
            self.tracker.setOfflineMode()
            self.tracker.sendCommand("clear_screen 0")
            err = self.tracker.doDriftCorrect(int(self.fix_el_x), int(self.fix_el_y), 1, 1)
            if err == pylink.ESC_KEY:
                pass  # operator pressed escape; continue
        except RuntimeError as e:
            print("Drift correct error:", e)

    def start_trial(self, trial_id: int):
        if not self.is_active():
            return
        self.tracker.setOfflineMode()
        self.tracker.sendCommand("clear_screen 0")
        self.tracker.sendMessage("TRIALID {:d}".format(trial_id))
        self.tracker.sendCommand("record_status_message 'Trial {:d}'".format(trial_id))
        try:
            file_rec = 1 if self.log_edf else 0
            self.tracker.startRecording(file_rec, file_rec, 1, 1)
        except RuntimeError as e:
            print("startRecording error:", e)
            return
        pylink.pumpDelay(50)

        # Resolve which eye to read.
        eu = self.tracker.eyeAvailable()
        if eu == 2:
            eu = 0   # binocular: prefer left
        self.eye_used = eu
        if eu == 1:
            self.tracker.sendMessage("EYE_USED 1 RIGHT")
        else:
            self.tracker.sendMessage("EYE_USED 0 LEFT")
        self.tracker.sendMessage("TRIAL_START {:d}".format(trial_id))

    def stop_trial(self, trial_id: int, result_code: int = 0):
        if not self.is_active():
            return
        try:
            self.tracker.sendMessage("TRIAL_END {:d}".format(trial_id))
            pylink.pumpDelay(50)
            self.tracker.stopRecording()
            self.tracker.sendMessage("TRIAL_RESULT {:d}".format(result_code))
        except Exception as e:
            print("stop_trial error:", e)

    def send_message(self, msg: str):
        if not self.is_active():
            return
        try:
            self.tracker.sendMessage(msg)
        except Exception:
            pass

    def get_gaze(self, prev_sample) -> Tuple[bool, Any]:
        """Read the latest sample. Returns (in_window, this_sample).

        in_window is True if the gaze is valid AND inside the fixation window.
        Returns (False, prev_sample) if no new sample is available.
        Blinks / missing data → in_window=False.
        """
        if not self.is_active():
            return True, prev_sample   # no enforcement → always pass
        try:
            new_sample = self.tracker.getNewestSample()
        except Exception:
            return False, prev_sample
        if new_sample is None:
            return False, prev_sample
        if prev_sample is not None and new_sample.getTime() == prev_sample.getTime():
            return None, prev_sample   # no NEW sample → caller should hold previous decision

        if self.eye_used == 1 and new_sample.isRightSample():
            gx, gy = new_sample.getRightEye().getGaze()
        elif self.eye_used == 0 and new_sample.isLeftSample():
            gx, gy = new_sample.getLeftEye().getGaze()
        else:
            return False, new_sample

        miss = EYE["missing_sentinel"]
        if gx == miss or gy == miss:
            return False, new_sample
        dist = math.hypot(gx - self.fix_el_x, gy - self.fix_el_y)
        return (dist <= self.fix_win_px), new_sample

    def close(self, output_dir: str, base_name: str):
        if not self.is_active():
            return
        try:
            if self.tracker.isConnected():
                if self.tracker.isRecording() == pylink.TRIAL_OK:
                    pylink.pumpDelay(100)
                    self.tracker.stopRecording()
                self.tracker.setOfflineMode()
                self.tracker.sendCommand("clear_screen 0")
                pylink.msecDelay(500)
                try:
                    self.tracker.closeDataFile()
                except Exception as e:
                    print("EDF close error:", e)
                if self.log_edf:
                    local_edf = os.path.join(output_dir, base_name + ".EDF")
                    try:
                        self.tracker.receiveDataFile(self.edf_filename, local_edf)
                        print("EDF saved to:", local_edf)
                    except RuntimeError as e:
                        print("EDF transfer error:", e)
                self.tracker.close()
        except Exception as e:
            print("EyeLink close error:", e)


# =========================
# Motion engine — one bar, optionally fixation-monitored
# =========================

def play_one_bar_motion(
    win,
    fixation,
    bar,
    sanity_circle,
    positions: List[float],
    bar_y: float,
    on_frames: int,
    off_frames: int,
    eyetracker: EyeLinkSession,
    pre_offset_enforce_frames: int,
    fix_break_consec_threshold: int,
    sanity_mode: bool = False,
) -> Dict[str, Any]:
    """Play one trial of one-bar discrete motion.

    Bar appears at positions[i] for `on_frames` frames, then blank for
    `off_frames` frames; sequence ends after the on-phase of positions[-1]
    (no trailing off-phase). Vanish moment = end of last on-phase.

    Fixation enforcement is active only in the last `pre_offset_enforce_frames`
    of motion. If gaze stays outside the fixation window (or sample is invalid)
    for `fix_break_consec_threshold` consecutive frames during the monitored
    period, motion is aborted and 'fixation_break' is returned.

    Returns a dict with status ∈ {"ok", "fixation_break"} and timing metadata.
    """
    consec_invalid = 0
    n_frames_displayed = 0
    motion_frame_idx = 0
    prev_sample = None
    last_in_window = True
    flashed_positions: List[float] = []

    n_pos = len(positions)
    motion_total_frames = n_pos * on_frames + (n_pos - 1) * off_frames
    enforce_start_frame = max(0, motion_total_frames - int(pre_offset_enforce_frames))
    for i, x in enumerate(positions):
        if sanity_mode:
            flashed_positions.append(x)
        else:
            bar.pos = (x, bar_y)

        # On-phase
        for _ in range(on_frames):
            if sanity_mode:
                draw_accumulated_flashes(fixation, bar, sanity_circle, flashed_positions, bar_y)
            else:
                fixation.draw()
                bar.draw()
            safe_flip(win)
            n_frames_displayed += 1

            if eyetracker.is_active() and motion_frame_idx >= enforce_start_frame:
                in_win, sample = eyetracker.get_gaze(prev_sample)
                if in_win is None:
                    in_win = last_in_window   # no new sample yet → hold previous decision
                else:
                    last_in_window = in_win
                prev_sample = sample
                if not in_win:
                    consec_invalid += 1
                    if consec_invalid >= fix_break_consec_threshold:
                        eyetracker.send_message(
                            "FIX_BREAK pos={:d} mframe={:d}".format(i, motion_frame_idx)
                        )
                        return {
                            "status":             "fixation_break",
                            "n_frames_displayed": n_frames_displayed,
                            "broke_at_position":  i,
                            "broke_at_frame":     motion_frame_idx,
                        }
                else:
                    consec_invalid = 0
            motion_frame_idx += 1

        # Off-phase (suppressed after the last position)
        if i < n_pos - 1:
            for _ in range(off_frames):
                if sanity_mode:
                    draw_accumulated_flashes(fixation, bar, sanity_circle, flashed_positions, bar_y)
                else:
                    fixation.draw()
                    # bar not drawn — blank
                safe_flip(win)
                n_frames_displayed += 1

                if eyetracker.is_active() and motion_frame_idx >= enforce_start_frame:
                    in_win, sample = eyetracker.get_gaze(prev_sample)
                    if in_win is None:
                        in_win = last_in_window
                    else:
                        last_in_window = in_win
                    prev_sample = sample
                    if not in_win:
                        consec_invalid += 1
                        if consec_invalid >= fix_break_consec_threshold:
                            eyetracker.send_message(
                                "FIX_BREAK pos={:d} mframe={:d}".format(i, motion_frame_idx)
                            )
                            return {
                                "status":             "fixation_break",
                                "n_frames_displayed": n_frames_displayed,
                                "broke_at_position":  i,
                                "broke_at_frame":     motion_frame_idx,
                            }
                    else:
                        consec_invalid = 0
                motion_frame_idx += 1

    return {
        "status":             "ok",
        "n_frames_displayed": n_frames_displayed,
        "broke_at_position":  -1,
        "broke_at_frame":     -1,
    }


# =========================
# Staircase machinery (unchanged from Exp 1a, slightly cleaned)
# =========================

def init_two_tracks(seeds: Dict[str, float], steps, bounds) -> Dict[str, Dict]:
    tracks: Dict[str, Dict] = {}
    for tid in ("A", "B"):
        seed = float(seeds[tid])
        tracks[tid] = {
            "id": tid, "seed": seed, "level": seed,
            "dir_prev": 0, "trial": 0, "n_rev": 0,
            "n_rev_in_step": 0, "step_index": 0, "reversals": [], "finished": False,
        }
    return tracks


def choose_track(tracks: Dict[str, Dict]) -> Optional[str]:
    unfinished = [tid for tid, t in tracks.items() if not t["finished"]]
    if not unfinished:
        return None
    if len(unfinished) == 1:
        return unfinished[0]
    return random.choice(unfinished)


def update_track_after_response(t, dir_sign, steps, tighten_after_reversals, bounds):
    lowB, highB = bounds
    step_index_before = t["step_index"]
    step_before = float(steps[step_index_before])
    level_before = float(t["level"])
    rev_count_level = t["n_rev_in_step"]

    is_reversal = False
    rev_level = None
    if t["dir_prev"] != 0 and dir_sign != 0 and (dir_sign != t["dir_prev"]):
        is_reversal = True
        rev_level = level_before
        t["n_rev"] += 1
        t["n_rev_in_step"] += 1
        rev_count_level = t["n_rev_in_step"]
        if step_index_before < len(steps) - 1:
            if step_index_before < len(tighten_after_reversals):
                if t["n_rev_in_step"] >= tighten_after_reversals[step_index_before]:
                    t["step_index"] = min(t["step_index"] + 1, len(steps) - 1)
                    t["n_rev_in_step"] = 0
        else:
            if step_index_before < len(tighten_after_reversals):
                if t["n_rev_in_step"] >= tighten_after_reversals[step_index_before]:
                    t["finished"] = True

    proposed = level_before + dir_sign * step_before
    proposed = max(min(proposed, highB), lowB)
    moved = (proposed != level_before)
    t["level"] = proposed
    if moved:
        t["dir_prev"] = dir_sign

    step_after = float(steps[t["step_index"]])
    tightened = (step_after != step_before)
    rev_index = t["n_rev"] if is_reversal else 0
    if is_reversal:
        t["reversals"].append(rev_level)
    return is_reversal, rev_level, rev_index, step_before, step_after, tightened, rev_count_level


def block_finished(tracks, cell_trial, max_trials_per_condition, sc) -> bool:
    if len(sc["tighten_after_reversals"]) < len(sc["steps"]):
        for t in tracks.values():
            if t["n_rev"] >= sc["max_reversals_per_track"]:
                t["finished"] = True
    if cell_trial >= max_trials_per_condition:
        return True
    return all(t["finished"] for t in tracks.values())


# =========================
# Instructions / practice
# =========================

def show_session_intro(win, cb):
    show_text_screen(win, [
        "Welcome to the experiment!",
    ], key=RESPONSE_KEYS["continue"])


def _check_instruction_keys(key_list, continue_key, quit_key):
    keys = event.getKeys(keyList=key_list)
    if not keys:
        return None
    k = keys[0].lower()
    if k == quit_key:
        return "quit"
    if k == continue_key:
        return "continue"
    return None


def run_instruction_stage(
    win, fixation, bar, bar_ghost,
    *,
    instruction_text: str,
    probe_color: str,
    soa_info: Dict[str, float],
    d_pool: Tuple[float, ...],
    polar_positions: List[Tuple[float, float, float]],
    refresh_hz: int,
    pre_fix_frames: int,
    ret_frames: int,
    iti_frames: int,
    show_probe: bool,
    show_ghost: bool,
    fixed_probe_side: Optional[str] = None,
):
    continue_key = RESPONSE_KEYS["continue"].lower()
    quit_key = RESPONSE_KEYS["quit"].lower()
    key_list = [continue_key, quit_key]
    event.clearEvents(eventType="keyboard")
    activate_window(win)

    instruction_txt = visual.TextStim(
        win, text=instruction_text,
        pos=(0, 420), height=26, color="white",
        alignText="center", wrapWidth=1200,
    )

    bar_speed_pf = STIM["bar_speed_px_per_sec"] / refresh_hz
    probe_frames = max(int(round(0.6 * refresh_hz)), 1)

    while True:
        cx, cy, _ = random.choice(polar_positions)
        d_init = float(random.choice(d_pool))
        motion_dir = random.choice([-1, +1])

        traj = compute_trajectory(soa_info, d_init, bar_speed_pf)
        positions = positions_for_trial(traj, cx, motion_dir)
        bar_y = cy
        bar_vanish_x = positions[-1]

        if fixed_probe_side == "right":
            probe_x = bar_vanish_x + abs(STAIRCASE["seeds"]["A"])
        elif fixed_probe_side == "left":
            probe_x = bar_vanish_x - abs(STAIRCASE["seeds"]["A"])
        else:
            probe_x = bar_vanish_x + random.choice([-1, +1]) * abs(STAIRCASE["seeds"]["A"])

        bar.fillColor = STIM["fg_color"]
        bar.lineColor = STIM["fg_color"]

        for _ in range(pre_fix_frames):
            fixation.draw()
            instruction_txt.draw()
            safe_flip(win)
            key = _check_instruction_keys(key_list, continue_key, quit_key)
            if key:
                return key

        for i, x in enumerate(positions):
            bar.pos = (x, bar_y)
            for _ in range(soa_info["on_frames"]):
                fixation.draw()
                bar.draw()
                instruction_txt.draw()
                safe_flip(win)
                key = _check_instruction_keys(key_list, continue_key, quit_key)
                if key:
                    return key
            if i < len(positions) - 1:
                for _ in range(soa_info["off_frames"]):
                    fixation.draw()
                    instruction_txt.draw()
                    safe_flip(win)
                    key = _check_instruction_keys(key_list, continue_key, quit_key)
                    if key:
                        return key

        if show_probe or show_ghost:
            for _ in range(ret_frames):
                fixation.draw()
                instruction_txt.draw()
                safe_flip(win)
                key = _check_instruction_keys(key_list, continue_key, quit_key)
                if key:
                    return key
            if show_probe:
                bar.pos = (probe_x, bar_y)
                bar.fillColor = probe_color
                bar.lineColor = probe_color
            if show_ghost:
                bar_ghost.pos = (bar_vanish_x, bar_y)
            for _ in range(probe_frames):
                fixation.draw()
                if show_ghost:
                    bar_ghost.draw()
                if show_probe:
                    bar.draw()
                instruction_txt.draw()
                safe_flip(win)
                key = _check_instruction_keys(key_list, continue_key, quit_key)
                if key:
                    return key

        bar.fillColor = STIM["fg_color"]
        bar.lineColor = STIM["fg_color"]
        for _ in range(iti_frames):
            fixation.draw()
            instruction_txt.draw()
            safe_flip(win)
            key = _check_instruction_keys(key_list, continue_key, quit_key)
            if key:
                return key


def run_instructions(
    win, fixation, bar, bar_ghost,
    *,
    probe_color: str,
    soa_frame_table: Dict[str, Dict[str, float]],
    refresh_hz: int,
    pre_fix_frames: int,
    ret_frames: int,
    iti_frames: int,
):
    polar_positions = enumerate_polar_positions()
    stages = [
        {
            "text": "In each trial, you will see a moving bar.",
            "soa": "smooth",
            "show_probe": False,
            "show_ghost": False,
            "fixed_probe_side": None,
        },
        {
            "text": "After the moving bar disappear, a color bar will appear.",
            "soa": "smooth",
            "show_probe": True,
            "show_ghost": False,
            "fixed_probe_side": None,
        },
        {
            "text": "Your task is to judge the color bar's position relative to the moving bar's last position.",
            "soa": "smooth",
            "show_probe": True,
            "show_ghost": True,
            "fixed_probe_side": None,
        },
        {
            "text": "If the color bar is on the right of the moving bar's last position, press K.",
            "soa": "smooth",
            "show_probe": True,
            "show_ghost": True,
            "fixed_probe_side": "right",
        },
        {
            "text": "If the color bar is on the left of the moving bar's last position, press F.",
            "soa": "smooth",
            "show_probe": True,
            "show_ghost": True,
            "fixed_probe_side": "left",
        },
        {
            "text": "In some part of the experiment, the bar will move in flashes; the task is the same.",
            "soa": "soa_300",
            "show_probe": True,
            "show_ghost": True,
            "fixed_probe_side": None,
        },
        {
            "text": (
                "Please look at the dot in the center during the animation.\n"
                "Do not move your eye until the response stage."
            ),
            "soa": "soa_300",
            "show_probe": True,
            "show_ghost": True,
            "fixed_probe_side": None,
        },
    ]

    for stage in stages:
        soa_label = stage["soa"]
        soa_info = soa_frame_table[soa_label]
        result = run_instruction_stage(
            win, fixation, bar, bar_ghost,
            instruction_text=stage["text"],
            probe_color=probe_color,
            soa_info=soa_info,
            d_pool=D_INIT_POOLS[soa_label],
            polar_positions=polar_positions,
            refresh_hz=refresh_hz,
            pre_fix_frames=pre_fix_frames,
            ret_frames=ret_frames,
            iti_frames=iti_frames,
            show_probe=stage["show_probe"],
            show_ghost=stage["show_ghost"],
            fixed_probe_side=stage["fixed_probe_side"],
        )
        if result == "quit":
            return "quit"

    show_text_screen(win, [
        "Let's do some practice.",
    ], key=RESPONSE_KEYS["continue"])
    return "ok"


def show_block_intro(win, block_idx, n_blocks, soa_label, soa_info, probe_color):
    motion_txt = (
        "The bar will move continously."
        if soa_label == "smooth"
        else "The bar will move in flashes."
    )
    show_text_screen(win, [
        f"Block {block_idx + 1} of {n_blocks}",
        motion_txt,
        "Maintain fixation on the central dot until the response stage.",
        "Press SPACE to begin.",
    ], key=RESPONSE_KEYS["continue"])


def show_block_break(win, block_idx, n_blocks):
    show_text_screen(win, [
        f"End of block {block_idx + 1} of {n_blocks}.",
        "",
        "Take a short break.",
        "",
        "Press SPACE when ready to continue.",
    ], key=RESPONSE_KEYS["continue"])


def farewell(win):
    msg = visual.TextStim(
        win, text="End of experiment.\n\nThank you!",
        pos=(0, 0), height=32, color="white",
    )
    wait_for_keypress(win, ["space", "escape"], quit_key=RESPONSE_KEYS["quit"], draw_func=msg.draw)


# Optional brief practice block before the first SOA block.
def run_practice(win, fixation, bar, bar_ghost, sanity_circle, refresh_hz, pre_fix_frames, ret_frames, iti_frames,
                 probe_color, soa_frame_table, eyetracker, pre_offset_frames, post_offset_frames,
                 fix_break_consec_threshold, state, stair_viz=None):
    """Quick familiarisation: smooth and SOA-300 motion at moderate offsets, with feedback."""
    practice_blocks = [
        ("smooth", "Practice (smooth)"),
        ("soa_300", "Practice (flash)"),
    ]

    polar_positions = enumerate_polar_positions()
    bar_speed_pf = STIM["bar_speed_px_per_sec"] / refresh_hz

    practice_offset_seq_A = [60, 60, 45, 45, 30, 30, 15, 15]
    practice_offset_seq_B = [-60, -60, -45, -45, -30, -30, -15, -15]
    practice_trials_per_track = len(practice_offset_seq_A)
    practice_trials_total = practice_trials_per_track * 2
    bounds_low, bounds_high = STAIRCASE["bounds"]

    bar.fillColor = STIM["fg_color"]
    bar.lineColor = STIM["fg_color"]

    for block_idx, (practice_soa, practice_title) in enumerate(practice_blocks):
        show_text_screen(win, [
            f"Practice block {block_idx + 1} of {len(practice_blocks)}",
            practice_title,
            "Please wait for instruction.",
        ], key=RESPONSE_KEYS["continue"])

        soa_info = soa_frame_table[practice_soa]
        d_pool = D_INIT_POOLS[practice_soa]
        practice_tracks = {
            "A": {"index": 0, "count": 0},
            "B": {"index": 0, "count": 0},
        }
        practice_trial_index = 0

        if stair_viz is not None:
            stair_viz.reset(
                "Practice (staircase) - {}".format(practice_soa),
                (bounds_low, bounds_high),
                subtitle="A/B tracks, step=20 px",
            )

        while practice_trial_index < practice_trials_total:
            tid = "A" if (practice_trial_index % 2 == 0) else "B"
            if practice_tracks[tid]["count"] >= practice_trials_per_track:
                tid = "B" if tid == "A" else "A"

            idx = practice_tracks[tid]["index"]
            level = float(practice_offset_seq_A[idx] if tid == "A" else practice_offset_seq_B[idx])

            while True:
                state["global_trial"] += 1
                cx, cy, _ = random.choice(polar_positions)
                d_init = random.choice(d_pool)
                motion_dir = random.choice([-1, +1])

                traj = compute_trajectory(soa_info, d_init, bar_speed_pf)
                positions = positions_for_trial(traj, cx, motion_dir)
                bar_y = cy

                # Pre-fix
                for _ in range(pre_fix_frames):
                    fixation.draw()
                    safe_flip(win)

                # Motion with eye-tracking in practice
                bar.fillColor = STIM["fg_color"]
                bar.lineColor = STIM["fg_color"]
                eyetracker.start_trial(state["global_trial"])
                eyetracker.send_message("PRACTICE_TRIAL soa={}".format(practice_soa))
                eyetracker.send_message("MOTION_ONSET")
                outcome = play_one_bar_motion(
                    win, fixation, bar, sanity_circle, positions, bar_y,
                    on_frames=soa_info["on_frames"],
                    off_frames=soa_info["off_frames"],
                    eyetracker=eyetracker,
                    pre_offset_enforce_frames=pre_offset_frames,
                    fix_break_consec_threshold=fix_break_consec_threshold,
                    sanity_mode=False,
                )

                if outcome["status"] == "fixation_break":
                    eyetracker.send_message("MOTION_OFFSET fix_break")
                    eyetracker.stop_trial(state["global_trial"], result_code=1)
                    show_brief_message(win, "Fixation broken!",
                                        duration_sec=0.9, color="orange",
                                        refresh_hz=refresh_hz)
                    for _ in range(int(0.5 * refresh_hz)):
                        fixation.draw()
                        safe_flip(win)
                    continue

                eyetracker.send_message("MOTION_OFFSET ok")

                post_enforce = min(int(post_offset_frames), int(ret_frames))
                if post_enforce > 0:
                    post_outcome = enforce_fixation_for_frames(
                        win, fixation, eyetracker, post_enforce, fix_break_consec_threshold
                    )
                    if post_outcome["status"] == "fixation_break":
                        eyetracker.send_message("FIX_BREAK post_offset")
                        eyetracker.stop_trial(state["global_trial"], result_code=1)
                        show_brief_message(win, "Fixation broken!",
                                            duration_sec=0.9, color="orange",
                                            refresh_hz=refresh_hz)
                        for _ in range(int(0.5 * refresh_hz)):
                            fixation.draw()
                            safe_flip(win)
                        continue

                for _ in range(ret_frames - post_enforce):
                    fixation.draw()
                    safe_flip(win)

                # Probe
                vanish_x = positions[-1]
                probe_x = vanish_x + motion_dir * level
                bar.pos = (probe_x, bar_y)
                bar.fillColor = probe_color
                bar.lineColor = probe_color

                if SANITY["enabled"]:
                    bar_ghost.pos = (vanish_x, bar_y)

                target_txt = visual.TextStim(
                    win, text="Does the color bar appear on the left or right of the moving bar's last position?",
                    pos=(0, 380), height=24, color="white", wrapWidth=1200,
                )
                resp_hint = visual.TextStim(
                    win, text=f"[{RESPONSE_KEYS['left'].upper()}] Left    [{RESPONSE_KEYS['right'].upper()}] Right",
                    pos=(0, -420), height=26, color="white",
                )

                def _draw_probe():
                    fixation.draw()
                    target_txt.draw()
                    bar.draw()
                    if SANITY["enabled"]:
                        bar_ghost.draw()
                    resp_hint.draw()

                resp_right, _ = wait_for_lr_keypress(win, _draw_probe)
                if resp_right == "quit":
                    eyetracker.stop_trial(state["global_trial"], result_code=0)
                    return "quit"
                eyetracker.send_message("RESPONSE")
                eyetracker.stop_trial(state["global_trial"], result_code=0)

                resp_left = (not resp_right)

                if stair_viz is not None:
                    subtitle = "Practice trial {} (track {})".format(practice_trial_index + 1, tid)
                    stair_viz.add_point(tid, level, subtitle=subtitle)

                # Correct = probe LEFT of vanish if motion_dir==1 and offset<0, etc.
                physical_probe_right_of_vanish = (probe_x > vanish_x)
                resp_correct = (bool(resp_right) == physical_probe_right_of_vanish)

                fb = visual.TextStim(
                    win,
                    text=("Correct!" if resp_correct
                          else f"Incorrect — probe was {'right' if physical_probe_right_of_vanish else 'left'} of vanish"),
                    pos=(0, 0), height=36,
                    color=("lime" if resp_correct else "red"),
                    bold=True, wrapWidth=1200,
                )
                for _ in range(int(1.2 * refresh_hz)):
                    fb.draw()
                    safe_flip(win)

                # Reset bar color for next trial
                bar.fillColor = STIM["fg_color"]
                bar.lineColor = STIM["fg_color"]

                for _ in range(iti_frames):
                    fixation.draw()
                    safe_flip(win)

                # Fixed per-track offset sequence (independent of response)
                practice_tracks[tid]["index"] += 1
                practice_tracks[tid]["count"] += 1
                practice_trial_index += 1
                break

        if block_idx < len(practice_blocks) - 1:
            show_text_screen(win, [
                "Smooth motion practice complete.",
                "",
                "Next: Flash motion practice.",
                "",
                "Press SPACE to continue.",
            ], key=RESPONSE_KEYS["continue"])

    show_text_screen(win, [
        "Practice complete!",
        "",
        "The real trials will use a range of motion speeds.",
        "",
        "Press SPACE to continue.",
    ], key=RESPONSE_KEYS["continue"])


# =========================
# Settings dialog
# =========================

def run_settings_dialog():
    info = {"Participant": "", "Age": ""}
    if not gui.DlgFromDict(info, title="RM Experiment 1b (v0.1n)").OK:
        core.quit()

    participant = str(info.get("Participant", "") or "999")
    age         = str(info.get("Age", "") or "NA")

    _hz_options = ["144", "60", "100", "120", "240"]
    _hz_current = str(HARDWARE["refresh_hz"])
    if _hz_current in _hz_options:
        _hz_options.remove(_hz_current)
        _hz_options.insert(0, _hz_current)

    _eye_options = ["off", "dummy", "live"]
    _eye_current = EYE["mode"]
    if _eye_current in _eye_options:
        _eye_options.remove(_eye_current)
        _eye_options.insert(0, _eye_current)

    settings = OrderedDict([
        ("Refresh Rate (Hz)",            _hz_options),
        ("Retention Interval (ms)",      str(int(TIMING["retention_sec"] * 1000))),
        ("Bar Speed (px/s)",             str(int(STIM["bar_speed_px_per_sec"]))),
        ("EyeLink Mode",                 _eye_options),
        ("Log EDF File",                 EYE["log_edf"]),
        ("Fixation Window (deg)",        str(EYE["fixation_window_deg"])),
        ("Fix Break Consec Frames",      str(EYE["fix_break_consec_frames"])),
        ("Run Instruction Block",        True),
        ("Run Practice Block",           True),
        ("Sanity Check Mode",            SANITY["enabled"]),
        ("Write Reversal CSV",           False),
        ("Fullscreen",                   HARDWARE["fullscreen"]),
        ("Mac Safe Flip",                HARDWARE["mac_safe_flip"]),
        ("DEMO Mode",                    DEMO["enabled"]),
    ])
    if not gui.DlgFromDict(settings, title="Experiment Settings (v0.1n)", sortKeys=False).OK:
        core.quit()

    HARDWARE["refresh_hz"]            = int(settings["Refresh Rate (Hz)"])
    TIMING["retention_sec"]           = float(settings["Retention Interval (ms)"]) / 1000.0
    STIM["bar_speed_px_per_sec"]      = float(settings["Bar Speed (px/s)"])
    EYE["mode"]                       = str(settings["EyeLink Mode"])
    EYE["log_edf"]                     = bool(settings["Log EDF File"])
    EYE["fixation_window_deg"]        = float(settings["Fixation Window (deg)"])
    EYE["fix_break_consec_frames"]    = int(settings["Fix Break Consec Frames"])
    SANITY["enabled"]                  = bool(settings["Sanity Check Mode"])
    HARDWARE["fullscreen"]            = bool(settings["Fullscreen"])
    HARDWARE["mac_safe_flip"]         = bool(settings["Mac Safe Flip"])
    DEMO["enabled"]                   = bool(settings["DEMO Mode"])

    flags = {
        "run_instructions":  bool(settings["Run Instruction Block"]),
        "run_practice":       bool(settings["Run Practice Block"]),
        "write_reversal_csv": bool(settings["Write Reversal CSV"]),
    }

    # Sanity: if EyeLink Mode != off but pylink is unavailable, downgrade.
    if EYE["mode"] != "off" and not HAS_PYLINK:
        print("WARNING: pylink not available — forcing EyeLink Mode to 'off'. ({})".format(_PYLINK_IMPORT_ERROR))
        EYE["mode"] = "off"

    return participant, age, flags


# =========================
# Trial loop for one SOA block
# =========================

def run_one_soa_block(
    win, fixation, bar, bar_ghost,
    sanity_circle,
    *,
    soa_label: str,
    soa_info: Dict[str, float],
    block_idx: int,
    n_blocks: int,
    probe_color: str,
    cb: Dict[str, Any],
    eyetracker: EyeLinkSession,
    main_writer, rev_writer,
    refresh_hz: int,
    pre_fix_frames: int,
    ret_frames: int,
    iti_frames: int,
    pre_offset_frames: int,
    post_offset_frames: int,
    fix_break_consec_threshold: int,
    sanity_mode: bool,
    sc: Dict,
    state: Dict,
    exp_date: str, exp_time: str,
    participant: str, age: str,
    demo_mode: int,
    stair_viz=None,
):
    """Run one SOA block, writing results into main_writer (and rev_writer if active)."""

    show_block_intro(win, block_idx, n_blocks, soa_label, soa_info, probe_color)

    if stair_viz is not None:
        title = "Block {} / {} ({})".format(block_idx + 1, n_blocks, soa_label)
        stair_viz.reset(title, sc["bounds"], subtitle="Probe color: {}".format(probe_color))

    # Drift correction at block boundary
    eyetracker.drift_correct()

    polar_positions = enumerate_polar_positions()
    bar_speed_pf = STIM["bar_speed_px_per_sec"] / refresh_hz

    tracks = init_two_tracks(sc["seeds"], sc["steps"], sc["bounds"])
    cell_trial = 0           # successful trials only
    block_trial_attempt = 0  # all attempts (including fix breaks)
    d_pool = D_INIT_POOLS[soa_label]

    on_frames  = soa_info["on_frames"]
    off_frames = soa_info["off_frames"]

    target_txt = visual.TextStim(
        win, text="Does the color bar appear on the left or right of the moving bar's last position?",
        pos=(0, 380), height=24, color="white", wrapWidth=1200,
    )
    resp_hint = visual.TextStim(
        win,
        text=f"[{RESPONSE_KEYS['left'].upper()}] Left    [{RESPONSE_KEYS['right'].upper()}] Right",
        pos=(0, -420), height=26, color="white",
    )

    while not block_finished(tracks, cell_trial, sc["max_trials_per_condition"], sc):
        tid = choose_track(tracks)
        if tid is None:
            break
        t = tracks[tid]

        level = float(t["level"])
        fixation_break_count = 0

        # Inner retry loop: re-roll geometry until fixation is maintained.
        while True:
            block_trial_attempt += 1
            state["global_trial"] += 1

            cx, cy, polar_deg = random.choice(polar_positions)
            d_init_nominal    = float(random.choice(d_pool))
            motion_dir        = random.choice([-1, +1])
            start_side        = "left" if motion_dir == 1 else "right"
            traj              = compute_trajectory(soa_info, d_init_nominal, bar_speed_pf)
            positions         = positions_for_trial(traj, cx, motion_dir)
            bar_y             = cy
            bar_start_x       = positions[0]
            bar_vanish_x      = positions[-1]

            # Reset bar color (motion bar is white)
            bar.fillColor = STIM["fg_color"]
            bar.lineColor = STIM["fg_color"]

            # Pre-fix
            for _ in range(pre_fix_frames):
                fixation.draw()
                safe_flip(win)

            # Begin EDF recording for this attempt
            eyetracker.start_trial(state["global_trial"])
            eyetracker.send_message(
                "STIM cx={:.2f} cy={:.2f} d_init={:.1f} dir={:d} soa={}".format(
                    cx, cy, d_init_nominal, motion_dir, soa_label
                )
            )
            eyetracker.send_message("MOTION_ONSET")

            # Motion phase
            event.clearEvents(eventType="keyboard")
            outcome = play_one_bar_motion(
                win, fixation, bar, sanity_circle, positions, bar_y,
                on_frames=on_frames, off_frames=off_frames,
                eyetracker=eyetracker,
                pre_offset_enforce_frames=pre_offset_frames,
                fix_break_consec_threshold=fix_break_consec_threshold,
                sanity_mode=sanity_mode,
            )

            # ---------- Fixation break path ----------
            if outcome["status"] == "fixation_break":
                eyetracker.send_message("MOTION_OFFSET fix_break")
                eyetracker.stop_trial(state["global_trial"], result_code=1)

                step_index_before = t["step_index"]
                step_level = step_index_before + 1
                step_size_px = float(sc["steps"][step_index_before])
                rev_count_level = t["n_rev_in_step"]

                _write_main_row(
                    main_writer,
                    exp_date=exp_date, exp_time=exp_time,
                    participant=participant, age=age,
                    demo_mode=demo_mode, eye_mode=EYE["mode"],
                    block_idx=block_idx, soa_label=soa_label,
                    cb=cb, probe_color=probe_color,
                    soa_info=soa_info, traj=traj,
                    d_init_nominal=d_init_nominal,
                    global_trial=state["global_trial"],
                    block_trial=cell_trial + 1,            # would-be successful trial number
                    block_trial_attempt=block_trial_attempt,
                    staircase_id=tid, t=t, sc=sc,
                    trial_outcome="fixation_break",
                    fixation_break_count=fixation_break_count + 1,
                    polar_deg=polar_deg, cx=cx, cy=cy, bar_y=bar_y,
                    start_side=start_side, motion_dir=motion_dir,
                    bar_start_x=bar_start_x, bar_vanish_x=bar_vanish_x,
                    refresh_hz=refresh_hz, pre_fix_frames=pre_fix_frames,
                    ret_frames=ret_frames, iti_frames=iti_frames,
                    bar_speed_pf=bar_speed_pf,
                    n_frames_displayed=outcome["n_frames_displayed"],
                    level_before=level,
                    step_level=step_level,
                    step_size_px=step_size_px,
                    step_actual=None,
                    probe_offset_motion=None, probe_x=None,
                    x_clean=None, probe_jitter=None, x_presented=None,
                    resp=None,
                    update_dir=0,
                    is_reversal=False, rev_index=0, rev_level=None,
                    step_size_before=float(sc["steps"][t["step_index"]]),
                    step_size_after=float(sc["steps"][t["step_index"]]),
                    tightened_step=False,
                    level_after=level,
                    n_reversals=t["n_rev"],
                    rev_count_level=rev_count_level,
                    fix_break_at_position=outcome["broke_at_position"],
                    fix_break_at_motion_frame=outcome["broke_at_frame"],
                )

                fixation_break_count += 1
                # Brief feedback then retry
                show_brief_message(win, "Fixation broken!",
                                    duration_sec=0.9, color="orange",
                                    refresh_hz=refresh_hz)
                # Brief blank fixation period
                for _ in range(int(0.5 * refresh_hz)):
                    fixation.draw()
                    safe_flip(win)
                continue   # retry without advancing the staircase

            # ---------- Successful motion → retention + probe ----------
            eyetracker.send_message("MOTION_OFFSET ok")
            post_enforce = min(int(post_offset_frames), int(ret_frames))
            if post_enforce > 0:
                post_outcome = enforce_fixation_for_frames(
                    win, fixation, eyetracker, post_enforce, fix_break_consec_threshold
                )
                if post_outcome["status"] == "fixation_break":
                    eyetracker.send_message("FIX_BREAK post_offset")
                    eyetracker.stop_trial(state["global_trial"], result_code=1)

                    step_index_before = t["step_index"]
                    step_level = step_index_before + 1
                    step_size_px = float(sc["steps"][step_index_before])
                    rev_count_level = t["n_rev_in_step"]

                    _write_main_row(
                        main_writer,
                        exp_date=exp_date, exp_time=exp_time,
                        participant=participant, age=age,
                        demo_mode=demo_mode, eye_mode=EYE["mode"],
                        block_idx=block_idx, soa_label=soa_label,
                        cb=cb, probe_color=probe_color,
                        soa_info=soa_info, traj=traj,
                        d_init_nominal=d_init_nominal,
                        global_trial=state["global_trial"],
                        block_trial=cell_trial + 1,
                        block_trial_attempt=block_trial_attempt,
                        staircase_id=tid, t=t, sc=sc,
                        trial_outcome="fixation_break",
                        fixation_break_count=fixation_break_count + 1,
                        polar_deg=polar_deg, cx=cx, cy=cy, bar_y=bar_y,
                        start_side=start_side, motion_dir=motion_dir,
                        bar_start_x=bar_start_x, bar_vanish_x=bar_vanish_x,
                        refresh_hz=refresh_hz, pre_fix_frames=pre_fix_frames,
                        ret_frames=ret_frames, iti_frames=iti_frames,
                        bar_speed_pf=bar_speed_pf,
                        n_frames_displayed=outcome["n_frames_displayed"],
                        level_before=level,
                        step_level=step_level,
                        step_size_px=step_size_px,
                        step_actual=None,
                        probe_offset_motion=None, probe_x=None,
                        x_clean=None, probe_jitter=None, x_presented=None,
                        resp=None,
                        update_dir=0,
                        is_reversal=False, rev_index=0, rev_level=None,
                        step_size_before=float(sc["steps"][t["step_index"]]),
                        step_size_after=float(sc["steps"][t["step_index"]]),
                        tightened_step=False,
                        level_after=level,
                        n_reversals=t["n_rev"],
                        rev_count_level=rev_count_level,
                        fix_break_at_position=-1,
                        fix_break_at_motion_frame=traj["motion_total_frames"] + post_outcome["broke_at_frame"],
                    )

                    fixation_break_count += 1
                    show_brief_message(win, "Fixation broken!",
                                        duration_sec=0.9, color="orange",
                                        refresh_hz=refresh_hz)
                    for _ in range(int(0.5 * refresh_hz)):
                        fixation.draw()
                        safe_flip(win)
                    continue
            if sanity_mode:
                def _draw_frozen():
                    draw_accumulated_flashes(fixation, bar, sanity_circle, positions, bar_y)

                key = wait_for_keypress(
                    win,
                    [RESPONSE_KEYS["continue"]],
                    quit_key=RESPONSE_KEYS["quit"],
                    draw_func=_draw_frozen,
                )
                if key == "quit":
                    return "quit"
            else:
                for _ in range(ret_frames - post_enforce):
                    fixation.draw()
                    safe_flip(win)
            eyetracker.send_message("PROBE_ONSET")

            x_clean = bar_vanish_x + motion_dir * level   # offset along motion direction
            while True:
                probe_jitter = int(round(random.gauss(0.0, 2.0)))
                if -4 <= probe_jitter <= 4:
                    break
            x_presented = x_clean + probe_jitter
            step_actual = level + motion_dir * probe_jitter
            probe_x = x_presented
            bar.pos = (probe_x, bar_y)
            bar.fillColor = probe_color
            bar.lineColor = probe_color

            if sanity_mode:
                bar_ghost.pos = (bar_vanish_x, bar_y)

            def _draw_probe_response():
                fixation.draw()
                target_txt.draw()
                bar.draw()
                if sanity_mode:
                    bar_ghost.draw()
                resp_hint.draw()

            resp_right, rt_ms = wait_for_lr_keypress(win, _draw_probe_response)
            eyetracker.send_message("RESPONSE rt_ms={}".format(rt_ms))
            eyetracker.stop_trial(state["global_trial"], result_code=0)

            if resp_right == "quit":
                return "quit"

            resp_left = (not resp_right)
            resp_key  = RESPONSE_KEYS["right"] if resp_right else RESPONSE_KEYS["left"]

            # Forward = same direction as motion (motion_dir = +1 → "right" key = forward)
            if motion_dir == 1:
                resp_forward = bool(resp_right)
            else:
                resp_forward = bool(resp_left)
            resp_backward = (not resp_forward)

            # Staircase update: if perceived forward → reduce probe offset (push away from motion).
            dir_sign = -1 if resp_forward else +1
            level_before = level
            step_index_before = t["step_index"]
            step_level = step_index_before + 1
            step_size_px = float(sc["steps"][step_index_before])
            (is_rev, rev_level, rev_idx,
             step_before, step_after, tightened, rev_count_level) = update_track_after_response(
                t, dir_sign, sc["steps"], sc["tighten_after_reversals"], sc["bounds"]
            )
            t["trial"] += 1
            cell_trial += 1

            if stair_viz is not None:
                subtitle = "Trial {} (track {})".format(cell_trial, tid)
                stair_viz.add_point(tid, step_actual, subtitle=subtitle)

            # Reversal sidecar
            if rev_writer is not None and is_rev:
                rev_writer.writerow({
                    "exp_date": exp_date, "exp_time": exp_time,
                    "participant": participant, "age": age,
                    "demo_mode": demo_mode, "eye_mode": EYE["mode"],
                    "block_index": block_idx, "soa_label": soa_label, "task": "probe",
                    "staircase_id": tid,
                    "speed_px_per_sec": STIM["bar_speed_px_per_sec"],
                    "retention_interval_ms": int(round(TIMING["retention_sec"] * 1000.0)),
                    "track_trial_at_reversal": t["trial"],
                    "rev_index": rev_idx, "rev_level": rev_level,
                    "update_dir_that_caused": dir_sign,
                    "step_before": step_before, "step_after": step_after,
                    "bounds_low": sc["bounds"][0], "bounds_high": sc["bounds"][1],
                    "block_order_idx": cb["block_order_idx"],
                    "block_order_str": cb["block_order_str"],
                    "color_idx": cb["color_idx"], "probe_color": cb["probe_color"],
                })

            # Main log
            _write_main_row(
                main_writer,
                exp_date=exp_date, exp_time=exp_time,
                participant=participant, age=age,
                demo_mode=demo_mode, eye_mode=EYE["mode"],
                block_idx=block_idx, soa_label=soa_label,
                cb=cb, probe_color=probe_color,
                soa_info=soa_info, traj=traj,
                d_init_nominal=d_init_nominal,
                global_trial=state["global_trial"],
                block_trial=cell_trial,
                block_trial_attempt=block_trial_attempt,
                staircase_id=tid, t=t, sc=sc,
                trial_outcome="ok",
                fixation_break_count=fixation_break_count,
                polar_deg=polar_deg, cx=cx, cy=cy, bar_y=bar_y,
                start_side=start_side, motion_dir=motion_dir,
                bar_start_x=bar_start_x, bar_vanish_x=bar_vanish_x,
                refresh_hz=refresh_hz, pre_fix_frames=pre_fix_frames,
                ret_frames=ret_frames, iti_frames=iti_frames,
                bar_speed_pf=bar_speed_pf,
                n_frames_displayed=outcome["n_frames_displayed"],
                level_before=level_before,
                step_level=step_level,
                step_size_px=step_size_px,
                step_actual=step_actual,
                probe_offset_motion=level_before,    # by definition: probe offset along motion = level
                probe_x=probe_x,
                x_clean=x_clean, probe_jitter=probe_jitter, x_presented=x_presented,
                resp={
                    "left": resp_left, "right": resp_right, "key": resp_key, "rt_ms": rt_ms,
                    "forward": resp_forward, "backward": resp_backward,
                },
                update_dir=dir_sign,
                is_reversal=is_rev, rev_index=rev_idx, rev_level=rev_level,
                step_size_before=step_before, step_size_after=step_after,
                tightened_step=tightened,
                level_after=t["level"],
                n_reversals=t["n_rev"],
                rev_count_level=rev_count_level,
                fix_break_at_position=-1, fix_break_at_motion_frame=-1,
            )

            # Reset bar color and ITI
            bar.fillColor = STIM["fg_color"]
            bar.lineColor = STIM["fg_color"]
            for _ in range(iti_frames):
                fixation.draw()
                safe_flip(win)
            break   # exit retry loop, choose next track

    return "ok"


def _write_main_row(
    writer, *,
    exp_date, exp_time, participant, age, demo_mode, eye_mode,
    block_idx, soa_label, cb, probe_color,
    soa_info, traj, d_init_nominal,
    global_trial, block_trial, block_trial_attempt,
    staircase_id, t, sc,
    trial_outcome, fixation_break_count,
    polar_deg, cx, cy, bar_y,
    start_side, motion_dir, bar_start_x, bar_vanish_x,
    refresh_hz, pre_fix_frames, ret_frames, iti_frames,
    bar_speed_pf, n_frames_displayed,
    level_before, step_level, step_size_px, step_actual,
    probe_offset_motion, probe_x,
    x_clean, probe_jitter, x_presented,
    resp,                       # dict or None
    update_dir, is_reversal, rev_index, rev_level,
    step_size_before, step_size_after, tightened_step,
    level_after, n_reversals,
    rev_count_level,
    fix_break_at_position, fix_break_at_motion_frame,
):
    """Write one row of the main CSV. resp=None → fixation_break attempt."""

    motion_total_frames = traj["motion_total_frames"]
    frame_period_ms = 1000.0 / float(refresh_hz)
    motion_total_dur_ms = motion_total_frames * frame_period_ms

    if resp is not None:
        resp_left  = int(resp["left"])
        resp_right = int(resp["right"])
        resp_key   = resp["key"]
        rt_ms      = resp["rt_ms"]
        resp_forward  = int(resp["forward"])
        resp_backward = int(resp["backward"])
        x_motion  = float(probe_offset_motion) if probe_offset_motion is not None else ""
        y_forward = int(resp["forward"])
    else:
        resp_left = resp_right = ""
        resp_key = ""
        rt_ms = ""
        resp_forward = resp_backward = ""
        x_motion = ""
        y_forward = ""

    writer.writerow({
        "exp_date": exp_date, "exp_time": exp_time,
        "participant": participant, "age": age,
        "demo_mode": demo_mode, "eye_mode": eye_mode,

        "block_index": block_idx, "block_soa_label": soa_label,
        "block_order_idx": cb["block_order_idx"],
        "block_order_str": cb["block_order_str"],
        "color_idx": cb["color_idx"], "probe_color": cb["probe_color"],

        "soa_label": soa_label,
        "on_dur_ms_nominal":  soa_info["on_dur_ms_nominal"],
        "off_dur_ms_nominal": soa_info["off_dur_ms_nominal"],
        "on_frames":          soa_info["on_frames"],
        "off_frames":         soa_info["off_frames"],
        "soa_ms_actual":      round(soa_info["soa_ms_actual"], 4),
        "pos_step_px":        round(traj["pos_step"], 4),
        "n_positions":        traj["n_positions"],
        "motion_total_frames": motion_total_frames,
        "motion_total_dur_ms": round(motion_total_dur_ms, 3),
        "d_init_nominal":     d_init_nominal,
        "d_init_effective":   round(traj["d_init_effective"], 4),

        "task": "probe",
        "global_trial": global_trial,
        "block_trial": block_trial,
        "block_trial_attempt": block_trial_attempt,
        "staircase_id": staircase_id,
        "track_id": staircase_id,
        "track_trial":  t["trial"],
        "start_level_seed": t["seed"],
        "trial_outcome": trial_outcome,
        "fixation_break_count_for_trial": fixation_break_count,

        "radius_px": RADIUS_PX,
        "polar_angle_deg": polar_deg,
        "cx": round(cx, 3), "cy": round(cy, 3),
        "bar_y": round(bar_y, 3),
        "start_side": start_side,
        "motion_dir": motion_dir,
        "bar_start_x":  round(bar_start_x, 3),
        "bar_vanish_x": round(bar_vanish_x, 3),

        "refresh_hz": refresh_hz,
        "pre_fix_frames": pre_fix_frames,
        "retention_frames": ret_frames,
        "iti_frames": iti_frames,
        "bar_speed_px_per_frame": round(bar_speed_pf, 6),
        "speed_px_per_sec": STIM["bar_speed_px_per_sec"],
        "retention_interval_ms": int(round(TIMING["retention_sec"] * 1000.0)),
        "n_frames_displayed": n_frames_displayed,

        "level_before":         round(level_before, 4),
        "step_level":           ("" if step_level is None else step_level),
        "step_size_px":         ("" if step_size_px is None else step_size_px),
        "step_actual":          ("" if step_actual is None else round(step_actual, 4)),
        "probe_offset_motion":  ("" if probe_offset_motion is None else round(probe_offset_motion, 4)),
        "probe_x":              ("" if probe_x is None else round(probe_x, 3)),
        "x_clean":              ("" if x_clean is None else round(x_clean, 3)),
        "probe_jitter":         ("" if probe_jitter is None else int(probe_jitter)),
        "x_presented":          ("" if x_presented is None else round(x_presented, 3)),

        "resp_left": resp_left, "resp_right": resp_right,
        "resp_key": resp_key, "rt_ms": rt_ms,
        "resp_forward": resp_forward, "resp_backward": resp_backward,

        "update_dir": update_dir,
        "step_size_before": step_size_before,
        "step_size_after":  step_size_after,
        "tightened_step":   int(bool(tightened_step)),
        "level_after":      round(level_after, 4),
        "n_reversals":      n_reversals,
        "rev_index":        rev_index,
        "is_reversal":      int(bool(is_reversal)),
        "rev_level":        ("" if rev_level is None else round(rev_level, 4)),
        "rev_count_level":  ("" if rev_count_level is None else rev_count_level),
        "rev_count_total":  n_reversals,
        "is_reversal_this_trial": int(bool(is_reversal)),
        "bounds_low":       sc["bounds"][0],
        "bounds_high":      sc["bounds"][1],

        "fix_break_at_position":     fix_break_at_position,
        "fix_break_at_motion_frame": fix_break_at_motion_frame,

        "x_motion":  x_motion,
        "y_forward": y_forward,
    })


# =========================
# Main
# =========================

def run():
    # --- Settings ---
    participant, age, flags = run_settings_dialog()
    pid_index = get_pid_index(participant)
    cb        = get_counterbalance(pid_index)
    demo_mode = int(bool(DEMO.get("enabled", False)))
    sc        = get_staircase_params()

    # --- Window ---
    mon = monitors.Monitor(
        "myMonitor",
        width=HARDWARE["scr_width_cm"],
        distance=HARDWARE["viewing_dist_cm"],
    )
    win = visual.Window(
        size=HARDWARE["resolution"],
        fullscr=HARDWARE["fullscreen"],
        monitor=mon,
        winType="pyglet",
        units="pix",
        color=STIM["bg_color"],
        checkTiming=False,
    )
    viz_win = None
    stair_viz = None

    refresh = HARDWARE["refresh_hz"]
    STIM["bar_speed"] = STIM["bar_speed_px_per_sec"] / refresh
    pre_fix_frames    = int(round(TIMING["pre_fix_sec"] * refresh))
    iti_frames        = int(round(TIMING["iti_sec"] * refresh))
    ret_frames        = int(round(TIMING["retention_sec"] * refresh))
    pre_offset_frames = int(round(EYE["pre_offset_monitor_ms"] / 1000.0 * refresh))
    post_offset_frames = int(round(EYE["post_offset_monitor_ms"] / 1000.0 * refresh))
    fix_break_consec  = int(EYE["fix_break_consec_frames"])

    # --- File paths ---
    exp_date = datetime.now().strftime("%Y%m%d")
    exp_time = datetime.now().strftime("%H%M%S")
    base = "RMprobe1b_{}_{}_{}_v0_1n".format(participant, exp_date, exp_time)
    if demo_mode:
        base += "_DEMO"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "Exp1b_Data")
    os.makedirs(out_dir, exist_ok=True)
    f_main = os.path.join(out_dir, base + ".csv")
    f_rev  = os.path.join(out_dir, base + "_reversals.csv")
    f_meta = os.path.join(out_dir, base + "_meta.json")

    # --- SOA frame table ---
    soa_frame_table = compute_soa_frame_counts(refresh)

    # --- Meta JSON ---
    meta = {
        "exp_date": exp_date, "exp_time": exp_time,
        "participant": participant, "age": age, "pid_index": pid_index,
        "demo_mode": demo_mode,
        "sanity_check_mode": SANITY["enabled"],
        "counterbalance": cb,
        "hardware": dict(HARDWARE),
        "stim": {k: STIM[k] for k in STIM if k != "bar_speed"},
        "timing": dict(TIMING),
        "radius_px": RADIUS_PX, "n_polar_angles": N_POLAR_ANGLES,
        "soa_conditions": [{"label": l, "on_ms": on, "off_ms": off}
                            for (l, on, off) in SOA_CONDITIONS],
        "soa_frame_table": soa_frame_table,
        "d_init_pools": {k: list(v) for k, v in D_INIT_POOLS.items()},
        "staircase": {
            "seeds": dict(sc["seeds"]),
            "steps": list(sc["steps"]),
            "tighten_after_reversals": list(sc["tighten_after_reversals"]),
            "bounds": list(sc["bounds"]),
            "max_reversals_per_track": sc["max_reversals_per_track"],
            "max_trials_per_condition": sc["max_trials_per_condition"],
        },
        "eye": dict(EYE),
        "response_keys": dict(RESPONSE_KEYS),
        "block_order": cb["block_order"],
    }
    with open(f_meta, "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)

    # --- EyeLink session (open before window-graphics handoff) ---
    eyetracker = EyeLinkSession(
        mode=EYE["mode"],
        edf_basename=EYE["edf_basename"],
        host_ip=EYE["host_ip"],
        calibration_type=EYE["calibration_type"],
        log_edf=EYE["log_edf"],
    )
    try:
        eyetracker.open()
    except RuntimeError as e:
        print("EyeLink open() failed:", e)
        win.close()
        if viz_win is not None:
            viz_win.close()
        core.quit()

    eyetracker.setup_graphics(win, fix_window_deg=EYE["fixation_window_deg"])

    if eyetracker.is_active():
        show_text_screen(win, [
            "Eye tracker calibration",
            "",
            "Press Space -> Enter to start calibration",
            "",
            "Press Escape here when calibration is complete.",
        ], key=RESPONSE_KEYS["continue"])
        eyetracker.calibrate()

    # --- Stimulus objects ---
    fixation, bar, bar_ghost, sanity_circle = make_stim_objects(win)

    state = {"global_trial": 0}

    # --- Session intro ---
    show_session_intro(win, cb)

    viz_win = create_staircase_window()
    stair_viz = StaircaseVisualizer(viz_win) if viz_win is not None else None
    show_staircase_startup(viz_win)
    activate_window(win)

    # --- Optional instructions ---
    if flags["run_instructions"]:
        result = run_instructions(
            win, fixation, bar, bar_ghost,
            probe_color=cb["probe_color"],
            soa_frame_table=soa_frame_table,
            refresh_hz=refresh,
            pre_fix_frames=pre_fix_frames,
            ret_frames=ret_frames,
            iti_frames=iti_frames,
        )
        if result == "quit":
            win.close()
            if viz_win is not None:
                viz_win.close()
            core.quit()
            return

    # --- Optional practice ---
    if flags["run_practice"]:
        result = run_practice(
            win, fixation, bar, bar_ghost, sanity_circle, refresh,
            pre_fix_frames, ret_frames, iti_frames,
            cb["probe_color"], soa_frame_table,
            eyetracker, pre_offset_frames, post_offset_frames,
            fix_break_consec, state, stair_viz=stair_viz,
        )
        if result == "quit":
            win.close()
            if viz_win is not None:
                viz_win.close()
            core.quit()
            return

    # --- Open log files ---
    rev_ctx = open(f_rev, "w", newline="", encoding="utf-8") if flags["write_reversal_csv"] else nullcontext()

    with open(f_main, "w", newline="", encoding="utf-8") as fmain, rev_ctx as frev:
        main_writer = csv.DictWriter(fmain, fieldnames=MAIN_HEADERS)
        main_writer.writeheader()
        rev_writer = None
        if flags["write_reversal_csv"]:
            rev_writer = csv.DictWriter(frev, fieldnames=REV_HEADERS)
            rev_writer.writeheader()

        for block_idx, soa_label in enumerate(cb["block_order"]):
            soa_info = soa_frame_table[soa_label]

            result = run_one_soa_block(
                win, fixation, bar, bar_ghost, sanity_circle,
                soa_label=soa_label, soa_info=soa_info,
                block_idx=block_idx, n_blocks=len(cb["block_order"]),
                probe_color=cb["probe_color"], cb=cb,
                eyetracker=eyetracker,
                main_writer=main_writer, rev_writer=rev_writer,
                refresh_hz=refresh,
                pre_fix_frames=pre_fix_frames,
                ret_frames=ret_frames,
                iti_frames=iti_frames,
                pre_offset_frames=pre_offset_frames,
                post_offset_frames=post_offset_frames,
                fix_break_consec_threshold=fix_break_consec,
                sanity_mode=SANITY["enabled"],
                sc=sc, state=state,
                exp_date=exp_date, exp_time=exp_time,
                participant=participant, age=age,
                demo_mode=demo_mode,
                stair_viz=stair_viz,
            )
            fmain.flush()

            if result == "quit":
                break

            # Break between blocks (skip after the last block)
            if block_idx < len(cb["block_order"]) - 1:
                show_block_break(win, block_idx, len(cb["block_order"]))

    # --- Close down ---
    eyetracker.close(out_dir, base)
    farewell(win)
    win.close()
    if viz_win is not None:
        viz_win.close()
    core.quit()


if __name__ == "__main__":
    run()
