"""
MOT_pilot_comprehensive.py

Two-part experiment scaffold derived from MOT Circular / practice files.
COMPREHENSIVE VERSION with detailed results logging matching MOT Circular.py

Parts:
 - Part 1: Off-Fixation Standard / Varying Speed block
 - Part 2: Different Shapes (diamond, sine-circle, ellipse)

General rules enforced:
 - 2 objects per trial (1 target, 1 distractor)
 - Two interleaved staircases per condition, 25 trials each
 - Adaptive staircase on speed across a 1.5 to 2.2 rps range
 - 5 fixed 0.5 rps attention-check trials per condition
 - Part 1: 5 conditions total = 275 trials including attention checks
 - Part 2: 3 conditions = 165 trials including attention checks
 - 440 total trials including attention checks
 - Circular trajectories use radius 6 deg
 - Feedback sounds: correct / incorrect

Results logging includes: trial details, initial angles, reversal times, trial duration
"""

from psychopy import prefs
prefs.hardware['audioLib'] = ['pygame']
from psychopy import visual, core, event, sound, monitors, data
import numpy as np, random, time, os
from collections import deque
from math import pi, cos, sin
from pathlib import Path
import atexit

# Ensure data directory exists to match earlier messages and avoid warnings
data_dir = Path('dataRaw')
if not data_dir.exists():
	try:
		data_dir.mkdir(parents=True, exist_ok=True)
	except Exception:
		pass

# Ensure a default monitor is present so PsychoPy doesn't create a temporary one
DEFAULT_MONITOR_NAME = 'default_monitor'
if DEFAULT_MONITOR_NAME not in monitors.getAllMonitors():
	try:
		m = monitors.Monitor(DEFAULT_MONITOR_NAME)
		# Best-effort defaults; adjust if you have a specific monitor
		try:
			m.setSizePix((1920, 1080))
			m.setWidth(52)  # physical width in cm
		except Exception:
			pass
		try:
			m.save()
		except Exception:
			pass
	except Exception:
		pass


# -------------------- Experiment parameters (match reference files) --------------------
numRings = 1
radii = np.array([6.0])  # deg, ensure circular trajectories use 6 degrees
respRadius = radii[0]
units = 'deg'

refreshRate = 100.0
trialDurMin = 2
trackingExtraTime = 1.2
trackVariableIntervMax = 2.5
autoAdvance = os.environ.get('MOT_AUTO_ADVANCE', '0').strip().lower() in ('1', 'true', 'yes', 'y', 'on')
longerThanRefreshTolerance = 0.27
longFrameLimit = round(1000.0 / refreshRate * (1.0 + longerThanRefreshTolerance), 3)

# PsychoPy `StairHandler` uses `nUp` = number of consecutive *incorrect*
# responses required to step in one direction, and `nDown` = number of
# consecutive *correct* responses required to step in the opposite direction.
# We want a 1-up / 3-down rule (1 incorrect -> step, 3 correct -> step), so
# set `nUp=1` and `nDown=3`.
stair_nUp = 1
stair_nDown = 3
stair_stepSizes = [.3, .3, .2, .1, .1, .05]
stair_start = 1.85
stair_trials_per_staircase = 25
stair_trials_per_condition = stair_trials_per_staircase * 2
attention_check_trials_per_condition = 5
attention_check_speed = 0.2

stair_start_speed_by_index = {
	1: 0.7,
	2: 1.5,
}

PART1_TRIAL_SPECS = [
	{'condition': 'centred', 'motionRule': 'standard', 'stairKey': 'Part1|centred|standard'},
	{'condition': 'near_displaced', 'motionRule': 'standard', 'stairKey': 'Part1|near_displaced|standard'},
	{'condition': 'far_displaced', 'motionRule': 'standard', 'stairKey': 'Part1|far_displaced|standard'},
	{'condition': 'near_displaced', 'motionRule': 'varying', 'stairKey': 'Part1|near_displaced|varying'},
	{'condition': 'far_displaced', 'motionRule': 'varying', 'stairKey': 'Part1|far_displaced|varying'},
]

PART2_TRIAL_SPECS = [
	{'condition': 'diamond', 'motionRule': 'shape', 'stairKey': 'Part2|diamond'},
	{'condition': 'sine_circle', 'motionRule': 'shape', 'stairKey': 'Part2|sine_circle'},
	{'condition': 'ellipse', 'motionRule': 'shape', 'stairKey': 'Part2|ellipse'},
]


def stair_value_to_speed(stair_value):
	"""Use the StairHandler value directly as speed."""
	return stair_value


def display_speed_to_stair_value(display_speed):
	"""Convert a displayed speed back into the StairHandler value space."""
	return display_speed


def make_attention_check_trials(part_name, trial_specs):
	trials = []
	for spec in trial_specs:
		for rep in range(attention_check_trials_per_condition):
			trials.append({
				'part': part_name,
				'condition': spec['condition'],
				'motionRule': spec.get('motionRule', 'standard'),
				'conditionKey': spec['stairKey'],
				'trialKind': 'attention_check',
				'speed': attention_check_speed,
			})
	random.shuffle(trials)
	return trials


def make_practice_trials(part_name, trial_specs):
	"""Build practice trials at 1.2 rps, 2 per condition for Part1, 3 per condition for Part2."""
	practice_speed = 1.2
	trials = []
	
	if part_name == 'Part1':
		trials_per_condition = 2
	elif part_name == 'Part2':
		trials_per_condition = 3
	else:
		trials_per_condition = 2
	
	for spec in trial_specs:
		for rep in range(trials_per_condition):
			trials.append({
				'part': part_name,
				'condition': spec['condition'],
				'motionRule': spec.get('motionRule', 'standard'),
				'conditionKey': spec['stairKey'],
				'trialKind': 'practice',
				'speed': practice_speed,
			})
	random.shuffle(trials)
	return trials


def make_condition_staircase_states(part_name, trial_specs):
	condition_states = []
	for spec in trial_specs:
		condition_key = spec['stairKey']
		state = {
			'part': part_name,
			'condition': spec['condition'],
			'motionRule': spec.get('motionRule', 'standard'),
			'conditionKey': condition_key,
			'next_staircase': 1,
			'queues': {
				1: deque(),
				2: deque(),
			},
		}
		for staircase_index in (1, 2):
			stair_start_speed = stair_start_speed_by_index[staircase_index]
			stair_key = f"{condition_key}|stair{staircase_index}"
			for rep in range(stair_trials_per_staircase):
				state['queues'][staircase_index].append({
					'part': part_name,
					'condition': spec['condition'],
					'motionRule': spec.get('motionRule', 'standard'),
					'conditionKey': condition_key,
					'staircaseIndex': staircase_index,
					'stairStartSpeed': stair_start_speed,
					'stairKey': stair_key,
				})
		condition_states.append(state)
	return condition_states


def build_interleaved_stair_trials(part_name, trial_specs):
	condition_states = make_condition_staircase_states(part_name, trial_specs)
	state_by_condition_key = {state['conditionKey']: state for state in condition_states}

	ordered_trials = []

	opening_round = []
	for state in condition_states:
		opening_round.append(state['queues'][1].popleft())
		state['next_staircase'] = 2
	random.shuffle(opening_round)
	ordered_trials.extend(opening_round)

	while True:
		available_trials = []
		for state in condition_states:
			next_staircase = state['next_staircase']
			if state['queues'][next_staircase]:
				available_trials.append(state['queues'][next_staircase][0])
		if not available_trials:
			break

		chosen_trial = random.choice(available_trials)
		state = state_by_condition_key[chosen_trial['conditionKey']]
		next_staircase = state['next_staircase']
		trial = state['queues'][next_staircase].popleft()
		ordered_trials.append(trial)
		state['next_staircase'] = 2 if next_staircase == 1 else 1

	attention_checks = make_attention_check_trials(part_name, trial_specs)
	if attention_checks:
		stair_trial_count = len(ordered_trials)
		early_buffer = max(10, len(trial_specs))
		spaced_region = max(1, stair_trial_count - early_buffer)
		insert_after_slots = early_buffer + np.linspace(1, spaced_region, len(attention_checks), dtype=int)
		spaced_trials = []
		stair_idx = 0
		for insert_after, attention_trial in zip(insert_after_slots, attention_checks):
			while stair_idx < insert_after and stair_idx < stair_trial_count:
				spaced_trials.append(ordered_trials[stair_idx])
				stair_idx += 1
			spaced_trials.append(attention_trial)
		spaced_trials.extend(ordered_trials[stair_idx:])
		ordered_trials = spaced_trials

	return ordered_trials


def get_part_trial_specs(part_name):
	if part_name == 'Part1':
		return PART1_TRIAL_SPECS
	if part_name == 'Part2':
		return PART2_TRIAL_SPECS
	return []


def create_staircase_window():
	# Try to query available screens (pyglet) and place the visualizer on the external/second monitor
	try:
		screen_index = 1
		win_size = [1600, 900]
		try:
			import pyglet
			display = pyglet.canvas.get_display()
			screens = display.get_screens()
			# prefer the second screen if present, otherwise use primary
			if len(screens) > 1:
				target = screens[1]
				screen_index = 1
			else:
				target = screens[0]
				screen_index = 0
			# cap window to screen size
			width = getattr(target, 'width', win_size[0])
			height = getattr(target, 'height', win_size[1])
			win_size = [min(win_size[0], int(width)), min(win_size[1], int(height))]
		except Exception:
			# If pyglet not available or querying fails, fall back to sensible defaults
			screen_index = 1
			win_size = [1600, 900]

		print(f"Staircase window: target_screen={screen_index}, size={win_size}")
		return visual.Window(
			size=win_size,
			screen=screen_index,
			fullscr=False,
			winType='pyglet',
			units='pix',
			color=[-0.9, -0.9, -0.9],
			allowGUI=False,
			waitBlanking=False,
			autoLog=False,
			checkTiming=False,
		)
	except Exception as e:
		print('Staircase window unavailable (continuing without it):', e)
		return None


def show_staircase_startup(win, part_name=None):
	if win is None:
		return
	activate_window(win)
	msg = 'Staircase window ready\nWaiting for trials...'
	if part_name:
		msg = f'{part_name}\n' + msg
	startup = visual.TextStim(
		win,
		text=msg,
		pos=(0, 0),
		height=24,
		color='white',
		alignText='center',
	)
	startup.draw()
	win.flip()


def activate_window(win):
	"""Best-effort foreground activation for secondary windows."""
	try:
		handle = getattr(win, 'winHandle', None)
		if handle is not None and hasattr(handle, 'activate'):
			handle.activate()
	except Exception:
		pass


class StaircaseVisualizer:
	def __init__(self, win):
		self.win = win
		self.part_name = ''
		self.title = 'Staircase Tracks'
		self.subtitle = ''
		self.y_bounds = (0.4, 2.4)
		self.max_points_per_track = stair_trials_per_staircase
		self.max_attention_points = attention_check_trials_per_condition
		self.panel_defs = []
		self.panel_data = {}
		self.attention_data = {}
		self.panel_order = []
		self.panel_lookup = {}
		self.track_colors = {1: 'dodgerblue', 2: 'magenta'}
		self.attention_color = 'gold'

	def set_layout(self, part_name, trial_specs, title=None, subtitle=None, y_bounds=None):
		if self.win is None:
			return
		self.part_name = part_name
		self.title = title or f'{part_name} staircase visualizer'
		self.subtitle = subtitle or 'Higher rps at top | Lower rps at bottom'
		if y_bounds is not None:
			self.y_bounds = y_bounds
		self.panel_defs = []
		self.panel_data = {}
		self.attention_data = {}
		self.panel_order = []
		self.panel_lookup = {}
		for spec in trial_specs:
			condition_key = spec['stairKey']
			# Friendly display name for the condition (replace underscores, capitalize)
			display_condition = spec['condition'].replace('_', ' ').title()
			panel_title = display_condition
			if part_name == 'Part1':
				panel_title = f"{display_condition} / {spec.get('motionRule', 'standard').title()}"
			self.panel_defs.append({
				'conditionKey': condition_key,
				'label': panel_title,
				'condition': spec['condition'],
				'motionRule': spec.get('motionRule', 'standard'),
			})
			self.panel_order.append(condition_key)
			self.panel_lookup[condition_key] = len(self.panel_defs) - 1
			# fixed-size slot lists for each staircase track (None = no data yet)
			# each slot will hold either None or a dict {'val': float, 'kind': 'stair'|'attention'}
			self.panel_data[condition_key] = {1: [None] * self.max_points_per_track, 2: [None] * self.max_points_per_track}
			self.attention_data[condition_key] = [None] * self.max_attention_points
		self.draw()

	def add_point(self, condition_key, staircase_index, level, trial_num=None, kind='stair', subtitle=None):
		"""Add a data point for a given condition/staircase.
		If trial_num is provided it is treated as 1-based index and the value is
		stored in that slot. Otherwise the method fills the next free slot.
		"""
		if self.win is None:
			return
		val = float(level)
		entry = {'val': val, 'kind': kind}
		if kind == 'attention':
			attn = self.attention_data.get(condition_key)
			if attn is None:
				return
			if trial_num is not None:
				idx = int(trial_num) - 1
				if 0 <= idx < len(attn):
					attn[idx] = entry
				else:
					return
			else:
				for i in range(len(attn)):
					if attn[i] is None:
						attn[i] = entry
						break
		else:
			panel = self.panel_data.get(condition_key)
			if panel is None:
				return
			if staircase_index not in panel:
				return
			vlist = panel[staircase_index]
			if trial_num is not None:
				idx = int(trial_num) - 1
				if 0 <= idx < len(vlist):
					vlist[idx] = entry
				else:
					return
			else:
				for i in range(len(vlist)):
					if vlist[i] is None:
						vlist[i] = entry
						break
		if subtitle is not None:
			self.subtitle = subtitle
		# optional debug printing
		try:
			if os.environ.get('MOT_VIS_DEBUG', '0').strip() in ('1', 'true', 'yes'):
				print(f"VIS_DEBUG: add_point cond={condition_key} stair={staircase_index} trial_num={trial_num} kind={kind} val={val}")
		except Exception:
			pass
		self.draw()

	def _panel_grid(self):
		n_panels = max(len(self.panel_order), 1)
		if self.part_name == 'Part1':
			cols, rows = 3, 2
		elif self.part_name == 'Part2':
			cols, rows = 3, 1
		else:
			cols = 2 if n_panels > 3 else n_panels
			rows = int(np.ceil(n_panels / float(cols)))
		return cols, rows

	def draw(self):
		if self.win is None:
			return

		w, h = self.win.size
		self.win.color = [-0.9, -0.9, -0.9]
		cols, rows = self._panel_grid()

		outer_x = 34
		outer_top = 76
		outer_bottom = 34
		gap_x = 20
		gap_y = 24
		panel_w = (w - 2 * outer_x - (cols - 1) * gap_x) / float(cols)
		panel_h = (h - outer_top - outer_bottom - (rows - 1) * gap_y) / float(rows)
		plot_pad_x = 34
		plot_pad_y = 30
		low, high = self.y_bounds
		span = high - low if (high - low) != 0 else 1.0

		title = visual.TextStim(self.win, text=self.title, pos=(0, h / 2 - 20), height=26, color='white')
		subtitle = visual.TextStim(self.win, text=self.subtitle, pos=(0, h / 2 - 50), height=17, color='white')
		title.draw()
		subtitle.draw()

		for idx, condition_key in enumerate(self.panel_order):
			panel_def = self.panel_defs[idx]
			col = idx % cols
			row = idx // cols
			left = -w / 2 + outer_x + col * (panel_w + gap_x)
			top = h / 2 - outer_top - row * (panel_h + gap_y)
			bottom = top - panel_h
			center_x = left + panel_w / 2.0
			center_y = bottom + panel_h / 2.0

			panel_box = visual.Rect(
				self.win,
				width=panel_w,
				height=panel_h,
				pos=(center_x, center_y),
				lineColor='white',
				fillColor=None,
			)
			panel_box.draw()

			plot_left = left + plot_pad_x
			plot_right = left + panel_w - plot_pad_x
			plot_bottom = bottom + plot_pad_y
			plot_top = bottom + panel_h - plot_pad_y
			plot_width = plot_right - plot_left
			plot_height = plot_top - plot_bottom

			# Frame and y-scale guide.
			visual.Line(self.win, start=(plot_left, plot_bottom), end=(plot_left, plot_top), lineColor='gray').draw()
			visual.Line(self.win, start=(plot_left, plot_bottom), end=(plot_right, plot_bottom), lineColor='gray').draw()

			label = visual.TextStim(
				self.win,
				text=panel_def['label'],
				pos=(center_x, top - 12),
				height=16,
				color='white',
				alignText='center',
			)
			label.draw()

			# draw colored swatches for the two staircases so each panel clearly
			# represents a single condition with two staircase colours
			swatch_x = left + 12
			swatch_y = bottom + 14
			swatch_gap = 60
			visual.Rect(self.win, width=10, height=10, pos=(swatch_x, swatch_y), fillColor=self.track_colors[1], lineColor=self.track_colors[1]).draw()
			visual.TextStim(self.win, text='stair1', pos=(swatch_x + 14, swatch_y), height=11, color='white', alignText='left').draw()
			visual.Rect(self.win, width=10, height=10, pos=(swatch_x + swatch_gap, swatch_y), fillColor=self.track_colors[2], lineColor=self.track_colors[2]).draw()
			visual.TextStim(self.win, text='stair2', pos=(swatch_x + swatch_gap + 14, swatch_y), height=11, color='white', alignText='left').draw()
			visual.Rect(self.win, width=10, height=10, pos=(swatch_x + 2 * swatch_gap, swatch_y), fillColor=self.attention_color, lineColor=self.attention_color).draw()
			visual.TextStim(self.win, text='attention', pos=(swatch_x + 2 * swatch_gap + 14, swatch_y), height=11, color='white', alignText='left').draw()

			# y-axis anchors so higher rps is visually higher.
			low_label = visual.TextStim(
				self.win,
				text=f'{low:.1f} rps',
				pos=(plot_left - 16, plot_bottom),
				height=11,
				color='white',
				alignText='right',
			)
			high_label = visual.TextStim(
				self.win,
				text=f'{high:.1f} rps',
				pos=(plot_left - 16, plot_top),
				height=11,
				color='white',
				alignText='right',
			)
			low_label.draw()
			high_label.draw()

			trial_left_label = visual.TextStim(
				self.win,
				text='1',
				pos=(plot_left, plot_bottom - 14),
				height=10,
				color='white',
			)
			trial_right_label = visual.TextStim(
				self.win,
				text=str(self.max_points_per_track),
				pos=(plot_right, plot_bottom - 14),
				height=10,
				color='white',
			)
			trial_left_label.draw()
			trial_right_label.draw()

			def _x_for_index(index):
				if self.max_points_per_track <= 1:
					return plot_left
				return plot_left + (plot_width * (index / float(self.max_points_per_track - 1)))

			def _x_for_attention_index(index):
				if self.max_attention_points <= 1:
					return plot_left
				return plot_left + (plot_width * (index / float(self.max_attention_points - 1)))

			def _y_for_speed(speed):
				return plot_bottom + ((speed - low) / span) * plot_height

			for staircase_index in (1, 2):
				values = self.panel_data[condition_key][staircase_index]
				# compute plotted points for fixed slots; skip None entries
				points = []
				for slot_idx, speed in enumerate(values):
					entry = speed
					if entry is None:
						continue
					val = entry.get('val') if isinstance(entry, dict) else float(entry)
					kind = entry.get('kind') if isinstance(entry, dict) else 'stair'
					x = _x_for_index(slot_idx)
					y = _y_for_speed(val)
					# collect stair points for line drawing, but keep all points for marker drawing
					if kind == 'stair':
						points.append((x, y))
					# draw marker for attention checks immediately (square/yellow), stair points drawn below
					if kind == 'attention':
						visual.Rect(self.win, width=6, height=6, pos=(x, y), fillColor='yellow', lineColor='yellow').draw()
				color = self.track_colors[staircase_index]
				if len(points) >= 2:
					for p1, p2 in zip(points[:-1], points[1:]):
						visual.Line(self.win, start=p1, end=p2, lineColor=color, lineWidth=2).draw()
				for point in points:
					visual.Circle(self.win, radius=3.5, pos=point, fillColor=color, lineColor=color).draw()

			attention_values = self.attention_data.get(condition_key, [])
			for attn_idx, entry in enumerate(attention_values):
				if entry is None:
					continue
				val = entry.get('val') if isinstance(entry, dict) else float(entry)
				x = _x_for_attention_index(attn_idx)
				y = _y_for_speed(val)
				visual.Circle(self.win, radius=3.8, pos=(x, y), fillColor=self.attention_color, lineColor=self.attention_color).draw()

		self.win.flip()

# Sounds
beep_correct = sound.Sound(value='C', secs=0.08)  # simple tone
beep_incorrect = sound.Sound(value='A', secs=0.12)

timeTillReversalMin = 0.5
timeTillReversalMax = 2.0
badTimingCushion = 0.3
def get_reversal_times(trial_duration_sec):
	reversal_times = []
	this_reversal_dur = trackingExtraTime
	while this_reversal_dur < trial_duration_sec + badTimingCushion:
		this_reversal_dur += np.random.uniform(timeTillReversalMin, timeTillReversalMax)
		reversal_times.append(this_reversal_dur)
	return reversal_times


# -------------------- Varying-speed modulation (copied style from MOT_Varying_speed) --------------------
ampTemporalRadiusModulation = 0.0
ampModulatnEachRingTemporalPhase = np.random.rand(numRings) * 2 * np.pi

def RFcontourCalcModulation(angle, freq, phase):
	return np.sin(angle * freq + phase)

RFcontourAmp = 0.0
RFcontourFreq = 2
RFcontourPhase = 0.0

def waveForm(type, speed, timeSeconds, numRing):
	if speed == 0 and ampTemporalRadiusModulation == 0:
		return 0
	else:
		periodOfRadiusModulation = 1.0 / speed if speed != 0 else 1.0
		modulatnPhaseRadians = timeSeconds / periodOfRadiusModulation * 2 * pi + ampModulatnEachRingTemporalPhase[numRing]
		if type == 'sin':
			return sin(modulatnPhaseRadians)
		elif type == 'sqrWave':
			ans = np.sign(sin(modulatnPhaseRadians))
			if ans == 0:
				ans = -1 + 2 * round(np.random.rand(1)[0])
			return ans
		else:
							# Only draw up to the most recent max_points_per_track values
							vals_to_plot = values[-self.max_points_per_track:]
							n_vals = len(vals_to_plot)
							# Start index so that x axis always spans 0..max_points_per_track-1
							start_idx = max(0, self.max_points_per_track - n_vals)
							for i, speed in enumerate(vals_to_plot):
								x_idx = start_idx + i
								x = _x_for_index(x_idx)
								y = _y_for_speed(speed)
								points.append((x, y))

def apply_visual_feedback(selected_idx, is_correct, x_pos, y_pos, blob_stim, fix):
	"""
	Flash selected object green (2x) if correct, red (2x) if incorrect.
	"""
	flash_color = [0, 1, 0] if is_correct else [1, -1, -1]  # green or red
	flash_duration = 0.1  # seconds per flash
	gap_duration = 0.1    # gap between flashes

	for flash_num in range(2):
		blob_stim.setFillColor(flash_color, log=False)
		blob_stim.setPos((x_pos, y_pos))
		fix.draw()
		blob_stim.draw()
		myWin.flip()
		core.wait(flash_duration)

		if flash_num == 0:
			fix.draw()
			myWin.flip()
			core.wait(gap_duration)


def apply_global_flash(is_correct, blobA, blobB, fix):
	"""Flash both objects green (correct) or bright orange (incorrect) twice.

	Parameters:
	- is_correct: bool
	- blobA/blobB: visual stimuli for the two objects
	- fix: fixation stimulus to draw underneath
	"""
	if is_correct:
		flash_col = (-1, 1, -1)  # green
	else:
		flash_col = (1, 0.5, -1)  # bright orange-ish
	flash_duration = 0.12
	gap_duration = 0.08
	for i in range(2):
		# set both blobs to flash color and draw
		blobA.setFillColor(flash_col, log=False)
		blobB.setFillColor(flash_col, log=False)
		fix.draw()
		# draw at their current positions (assume positions already set by caller)
		blobA.draw(); blobB.draw()
		myWin.flip()
		core.wait(flash_duration)
		# clear (draw fixation only) between flashes
		fix.draw(); myWin.flip(); core.wait(gap_duration)


def compute_mean_inv_rho(l, r, num_samples=360):
	"""
	Compute the mean of 1/rho over a full circle.
	Used to scale V for log-polar motion to ensure correct average revolution time.
	
	For log-polar speed equation phi_dot = (V/r) * rho, the time for one full revolution is:
	T = (r/V) * ∫[0, 2π] (1/rho(φ)) dφ = (r/V) * 2π * mean(1/rho)
	
	To get T = 1/speed, we need: V = speed * r * 2π * mean(1/rho)
	"""
	inv_rhos = []
	for i in range(num_samples):
		phi = 2 * np.pi * i / num_samples
		rho = np.sqrt(l*l + r*r + 2.0*l*r*np.cos(phi))
		if rho > 0:
			inv_rhos.append(1.0 / rho)
	return np.mean(inv_rhos) if inv_rhos else 1.0 / r


def rho_from_phi(phi, l, r):
	"""Distance from fixation to the moving object at angle phi."""
	return np.sqrt(l * l + r * r + 2.0 * l * r * np.cos(phi))


def phi_dot_log_eccentricity_base_speed(phi, base_rps, l, r):
	"""
	Varying-speed rule where base_rps is the speed the object would travel if the
	circle were centred on fixation.

	If l = 0, then rho = r and the rule reduces to ordinary circular motion:
	phi_dot = 2*pi*base_rps
	"""
	rho = rho_from_phi(phi, l, r)
	return 2 * np.pi * base_rps * (rho / r)


def update_phi_log_polar(phi, V, l, r, dt, direction=1.0):
	"""
	Update phi using constant log-polar speed equation.
	
	phi_dot = (V / r) * rho
	where rho = sqrt(l^2 + r^2 + 2*l*r*cos(phi))
	
	Parameters:
	- phi: current angle around circle centre
	- V: desired constant speed in log-polar space from fixation
	- l: distance from fixation to circle centre
	- r: radius of the circular path
	- dt: time step
	- direction: +1 or -1 for direction of motion
	"""
	rho = np.sqrt(l*l + r*r + 2.0*l*r*np.cos(phi))
	phi_dot = (V / r) * rho
	phi_dot = phi_dot * direction  # apply reversal direction
	phi_new = phi + phi_dot * dt
	phi_new = phi_new % (2 * np.pi)
	return phi_new


def get_xy_from_phi(phi, offset_xy, l, r):
	x = offset_xy[0] + r * cos(phi)
	y = offset_xy[1] + r * sin(phi)
	return x, y


# -------------------- Trajectory generators --------------------
def squareShape(angle):
	angle = angle % (2 * pi)
	p = angle / (2 * pi)
	halfSide = 1.0 / np.sqrt(2.0)
	u = p * 4.0
	if u < 1.0:
		x = halfSide
		y = -halfSide + (u * 2.0 * halfSide)
	elif u < 2.0:
		x = halfSide - ((u - 1.0) * 2.0 * halfSide)
		y = halfSide
	elif u < 3.0:
		x = -halfSide
		y = halfSide - ((u - 2.0) * 2.0 * halfSide)
	else:
		x = -halfSide + ((u - 3.0) * 2.0 * halfSide)
		y = -halfSide
	return x, y

def rotatePoint(x, y, angleRad):
	c = cos(angleRad)
	s = sin(angleRad)
	return x * c - y * s, x * s + y * c

def circle_xy(radius_deg, angle_rad, timeSeconds=None, speed=None, numRing=0):
	# allow optional temporal radius modulation
	r = radius_deg
	if timeSeconds is not None and speed is not None:
		rThis = r + waveForm('sin', speed, timeSeconds, numRing) * r * ampTemporalRadiusModulation
		rThis += r * RFcontourAmp * RFcontourCalcModulation(angle_rad, RFcontourFreq, RFcontourPhase)
	else:
		rThis = r
	return rThis * cos(angle_rad), rThis * sin(angle_rad)

def sine_circle_xy(radius_deg, angle_rad, amp=0.25):
	# match practice-file sine-wave circle geometry
	sineCircleAmplitudeDeg = 0.7
	sineCircleLobes = 12
	r = radius_deg + sineCircleAmplitudeDeg * np.sin(sineCircleLobes * angle_rad)
	return r * cos(angle_rad), r * sin(angle_rad)

def ellipse_xy(circleRadius, angle_rad):
	ellipseAspectRatio = 1.6
	ellipseRotationRad = pi / 4.0
	def ellipseCircumferenceApprox(a, b):
		h = ((a - b) ** 2) / ((a + b) ** 2)
		return pi * (a + b) * (1 + ((3 * h) / (10 + np.sqrt(4 - 3 * h))))
	def ellipseAxesMatchingCircleCircumference(circleRadius, aspectRatio):
		aRaw = circleRadius * aspectRatio
		bRaw = circleRadius / aspectRatio
		targetCirc = 2 * pi * circleRadius
		rawCirc = ellipseCircumferenceApprox(aRaw, bRaw)
		if rawCirc <= 0:
			return circleRadius, circleRadius
		scale = targetCirc / rawCirc
		return aRaw * scale, bRaw * scale
	a, b = ellipseAxesMatchingCircleCircumference(circleRadius, ellipseAspectRatio)
	xUnrot = a * cos(angle_rad)
	yUnrot = b * sin(angle_rad)
	return rotatePoint(xUnrot, yUnrot, ellipseRotationRad)

def diamond_xy(radius_deg, angle_rad):
	# match practice-file diamond trajectory geometry
	x, y = squareShape(angle_rad)
	x, y = rotatePoint(x, y, pi / 4.0)
	return radius_deg * x, radius_deg * y


def draw_trajectory(myWin, basicShape, radius_deg, cx, cy, num_points=120):
	"""
	Draw the actual trajectory (white line) that objects traveled during the trial.
	"""
	trajectory_points = []
	for i in range(num_points):
		angle_rad = 2 * np.pi * i / num_points
		
		if basicShape == 'circle':
			x, y = circle_xy(radius_deg, angle_rad)
		elif basicShape == 'diamond':
			x, y = diamond_xy(radius_deg, angle_rad)
		elif basicShape == 'sine_circle':
			x, y = sine_circle_xy(radius_deg, angle_rad)
		elif basicShape == 'ellipse':
			x, y = ellipse_xy(radius_deg, angle_rad)
		else:
			x, y = circle_xy(radius_deg, angle_rad)
		
		# Apply offset
		x += cx
		y += cy
		trajectory_points.append([x, y])
	
	# Draw trajectory as a line
	trajectory_line = visual.ShapeStim(
		myWin,
		vertices=trajectory_points,
		fillColor=None,
		lineColor=(1, 1, 1),
		lineWidth=2,
		closeShape=True
	)
	# trajectory_line.draw()


def build_session():
	part12_trials = build_interleaved_stair_trials('Part1', PART1_TRIAL_SPECS)
	part3_trials = build_interleaved_stair_trials('Part2', PART2_TRIAL_SPECS)

	# keep trials separate per part (they are run independently per instructions)
	return {'part1': part12_trials, 'part2': part3_trials}


def build_varying_staircases():
	staircases = {}
	for part_name, trial_specs in [('Part1', PART1_TRIAL_SPECS), ('Part2', PART2_TRIAL_SPECS)]:
		for spec in trial_specs:
			for staircase_index in (1, 2):
				stair_start_speed = stair_start_speed_by_index[staircase_index]
				staircases[f"{spec['stairKey']}|stair{staircase_index}"] = data.StairHandler(
					startVal=display_speed_to_stair_value(stair_start_speed),
					stepType='lin',
					stepSizes=stair_stepSizes,
					minVal=None,
					maxVal=2.5,
					nUp=stair_nUp,
					nDown=stair_nDown,
					applyInitialRule=False,
					nTrials=stair_trials_per_staircase,
					extraInfo={
						'part': part_name,
						'condition': spec['condition'],
						'motionRule': spec['motionRule'],
						'staircaseIndex': staircase_index,
					}
				)
	return staircases


def build_speed_staircases():
	"""Create a StairHandler for every stairKey (per-part, per-condition, per-motionRule).
	This is the canonical builder used by the run loop.
	Delegates to build_varying_staircases for the current experiment spec.
	"""
	return build_varying_staircases()


# -------------------- Minimal run loop for verification (non-optimized, non-graphical checks) --------------------
def run_checks_and_report(trials_by_part):
	# Check radii for circular trajectories
	circle_radius_ok = (abs(radii[0] - 6.0) < 1e-6)

	# Trial counts
	counts = {k: len(v) for k, v in trials_by_part.items()}
	expected_part_specs = {
		'part1': PART1_TRIAL_SPECS,
		'part2': PART2_TRIAL_SPECS,
	}
	expected_part_counts = {
		'part1': len(PART1_TRIAL_SPECS) * stair_trials_per_condition + len(PART1_TRIAL_SPECS) * attention_check_trials_per_condition,
		'part2': len(PART2_TRIAL_SPECS) * stair_trials_per_condition + len(PART2_TRIAL_SPECS) * attention_check_trials_per_condition,
	}

	# Counterbalancing: count speeds per part
	speed_counts = {}
	for pname, tlist in trials_by_part.items():
		sc = {}
		for t in tlist:
			# Some trials are staircase-driven and don't have a concrete 'speed' yet.
			# Count by numeric speed when present, otherwise by the staircase key.
			if 'speed' in t:
				key = t['speed']
			else:
				key = t.get('stairKey', 'staircase')
			sc[key] = sc.get(key, 0) + 1
		speed_counts[pname] = sc

	print('\n=== MOT_pilot verification report ===')
	print('Circular radii set to 6 deg:', circle_radius_ok)
	for pname in counts:
		expected = expected_part_counts[pname]
		print(f"{pname}: {counts[pname]} trials (expected {expected})")
		print('  speed distribution:', speed_counts[pname])

		stair_trials = [t for t in trials_by_part[pname] if t.get('trialKind') != 'attention_check']
		opening_round = stair_trials[:len(expected_part_specs[pname])]
		print('  opening round staircase indices:', [t['staircaseIndex'] for t in opening_round])
		print('  opening round condition keys:', [t['conditionKey'] for t in opening_round])

		for spec in expected_part_specs[pname]:
			condition_trials = [t for t in stair_trials if t.get('conditionKey') == spec['stairKey']]
			condition_sequence = [t['staircaseIndex'] for t in condition_trials]
			expected_sequence = [1, 2] * stair_trials_per_staircase
			if condition_sequence != expected_sequence:
				print('  sequence mismatch for', spec['stairKey'], '->', condition_sequence[:10], '...')

	total_trials = sum(counts.values())
	# compute expected total using the staircase split above
	expected_total = sum(expected_part_counts.values())
	print('Total trials across 2 parts =', total_trials, f'(expected {expected_total})')

	print('\nFeedback sounds available: correct/inaccurate assigned')
	print('beep_correct:', beep_correct)
	print('beep_incorrect:', beep_incorrect)

	# Quick assertions to flag any issue
	issues = []
	for pname in counts:
		expected = expected_part_counts[pname]
		if counts[pname] != expected:
			issues.append(f"{pname} has {counts[pname]} trials (expected {expected})")
	if total_trials != expected_total:
		issues.append('Total trials mismatch')

	if not circle_radius_ok:
		issues.append('Circular radius not 6 deg')

	if issues:
		print('\nIssues detected:')
		for it in issues:
			print('-', it)
	else:
		print('\nNo issues detected. Ready to run experiment implementation.')


if __name__ == '__main__':
	random.seed(int(time.time()))
	trials = build_session()
	run_checks_and_report(trials)
	parts_to_run = [part.strip().lower() for part in os.environ.get('MOT_PARTS', 'part1,part2').split(',') if part.strip()]
	trial_limit_raw = os.environ.get('MOT_TRIAL_LIMIT_PER_PART', '').strip()
	if trial_limit_raw:
		try:
			trial_limit = max(1, int(trial_limit_raw))
		except ValueError:
			raise ValueError(f"Invalid MOT_TRIAL_LIMIT_PER_PART value: {trial_limit_raw}")
		for part_key in ('part1', 'part2'):
			if part_key in trials:
				trials[part_key] = trials[part_key][:trial_limit]

	# -------------------- Data file setup --------------------
	subject = 'temp'
	session = 1
	timeAndDateStr = time.strftime("%d%b%Y_%H-%M", time.localtime())
	if os.path.isdir('.'+os.sep+'dataRaw'):
		dataDir='dataRaw'
	else:
		print('"dataRaw" directory does not exist, so saving data in present working directory')
		dataDir='.'
	datafileName = dataDir+'/'+subject+ '_' + str(session) + '_MOT_pilot_'+timeAndDateStr
	dataFile = open(datafileName+'.tsv', 'w')

	# -------------------- Full experiment run (display, timing, response collection) --------------------
	# Minimal window setup (matches practice units)
	try:
		myWin = visual.Window(size=(800, 600), units='deg', fullscr=False, color=(0, 0, 0))
	except Exception as e:
		print('Warning creating Window:', e)
		print('Creating temporary Monitor specification and retrying Window...')
		try:
			tempMon = monitors.Monitor('tempMonitor', width=38.0, distance=57.0)
			tempMon.setSizePix((800, 600))
			myWin = visual.Window(size=(800, 600), units='deg', fullscr=False, color=(0, 0, 0), monitor=tempMon, checkTiming=False)
		except Exception as e2:
			print('Failed to create Window with temporary monitor:', e2)
			raise

	# Stimuli
	fixation = visual.Circle(myWin, radius=0.2, fillColor=(1, 1, 1), lineColor=None)
	blobStim = visual.Circle(myWin, radius=0.6, fillColor=(1, -1, -1), lineColor=None)
	blobStim2 = visual.Circle(myWin, radius=0.6, fillColor=(-1, 1, -1), lineColor=None)
	stairWin = create_staircase_window()
	stairViz = StaircaseVisualizer(stairWin) if stairWin is not None else None
	show_staircase_startup(stairWin)

	# Trial timing
	cueFrames = int(refreshRate * trackingExtraTime)
	timingCheckFrames = cueFrames
	trialDurFrames = int(trialDurMin * refreshRate) + int(trackingExtraTime * refreshRate)
	trial_duration_sec = trialDurFrames / refreshRate

	# Eccentricities used for trajectory centers
	trajectory_center_x_deg = [0.0, 5.0, 10.0]
	trajectory_center_y_deg = [0.0, 0.0, 0.0]

	results = []
	identicalBlobColor = np.array([1, -1, -1])
	targetCueColor = np.array([1, 1, 1])
	# distractor during cue interval (red), flash colors for feedback
	distractorCueColor = np.array([1, -1, -1])
	trialClock = core.Clock()
	speed_staircases = build_speed_staircases()

	# Write comprehensive header to TSV file (matching MOT Circular format)
	# Column order is explicit for downstream analysis
	header_columns = [
		'trialnum', 'subject', 'session', 'part', 'condition', 'basicShape', 'numObjects', 'speed', 'motionRule', 'trialKind',
		'staircaseIndex', 'staircaseWithinCondition', 'stairKey', 'stairStartSpeed', 'stairValue',
		'initialAngle', 'initialOtherAngle', 'cueFrames', 'timingCheckFrames', 'correct', 'trialDurTotal', 'numTargets', 'whichIsTarget',
		'reversal_count', 'timingBlips', 'numLongFramesAfterFixation', 'numLongFramesAfterCue'
	]
	# add reversal columns
	for i in range(10):
		header_columns.append(f'reversal_{i}')
	header_line = '\t'.join(header_columns)
	print(header_line, file=dataFile)

	# Ensure data file and window get closed on exit
	def _close_resources():
		try:
			dataFile.close()
		except Exception:
			pass
		try:
			if 'myWin' in globals() and myWin is not None:
				myWin.close()
		except Exception:
			pass
		try:
			if 'stairWin' in globals() and stairWin is not None:
				stairWin.close()
		except Exception:
			pass
	atexit.register(_close_resources)

	def show_break_screen(message=None):
		"""Display a simple break screen and wait for the participant to press SPACE.
		If ESCAPE is pressed the experiment will quit.
		"""
		if message is None:
			message = 'Please take a short break. Press SPACE to continue.'
		# create a centered text stimulus for the break screen
		break_text = visual.TextStim(myWin, text=message, color=(1, 1, 1), height=0.8, wrapWidth=40)
		break_text.draw()
		myWin.flip()
		# wait until space is pressed
		while True:
			keys = event.waitKeys()
			if not keys:
				continue
			if 'space' in keys or 'spacebar' in keys:
				core.wait(0.1)
				break
			if 'escape' in keys:
				core.quit()

	def run_part(part_name, trial_list):
		print(f"Running {part_name} with {len(trial_list)} trials")
		
		# Build and run practice trials first
		part_specs = get_part_trial_specs(part_name)
		practice_trials = make_practice_trials(part_name, part_specs)
		if practice_trials:
			print(f"Running {len(practice_trials)} practice trials for {part_name}")
			show_break_screen(f"{part_name} Practice Trials\n\nYou will now do {len(practice_trials)} practice trials to familiarize yourself with the task.\n\nPress SPACE to begin.")
			# Run practice trials (they don't affect the staircase)
			for pt_idx, practice_trial in enumerate(practice_trials):
				# All practice code here (same as main loop but marked as practice)
				cond = practice_trial['condition']
				motion_rule = practice_trial.get('motionRule', 'standard')
				speed = practice_trial['speed']
				
				if part_name == 'Part1':
					if cond == 'centred':
						cx = trajectory_center_x_deg[0]
					elif cond == 'near_displaced':
						cx = trajectory_center_x_deg[1]
					elif cond == 'far_displaced':
						cx = trajectory_center_x_deg[2]
					cy = 0.0
					basicShape = 'circle'
				else:  # Part2 shapes
					cx = 0.0
					cy = 0.0
					basicShape = cond
				
				# Starting angles (two objects opposite)
				currTargetAngle = random.random() * 2 * pi
				distractorAngle = (currTargetAngle + pi) % (2 * pi)
				initialAngle = currTargetAngle
				initialOtherAngle = distractorAngle
				
				currPhi = [random.random() * 2 * pi, 0.0]
				currPhi[1] = (currPhi[0] + pi) % (2 * pi)
				direction = 1
				reversal_times = get_reversal_times(trial_duration_sec)
				next_reversal_idx = 0
				target_idx = 0
				trial_start_time = trialClock.getTime()
				ts = []
				
				# Frame loop (identical to main loop)
				for frameN in range(trialDurFrames):
					timeSec = frameN / refreshRate
					if part_name == 'Part1' and basicShape == 'circle' and motion_rule == 'varying':
						dt = 1.0 / refreshRate
						l = np.sqrt(cx * cx + cy * cy)
						base_rps = speed
						r = radii[0]
						
						phi_dot_0 = phi_dot_log_eccentricity_base_speed(currPhi[0], base_rps, l, r)
						currPhi[0] = (currPhi[0] + direction * phi_dot_0 * dt) % (2 * np.pi)
						phi_dot_1 = phi_dot_log_eccentricity_base_speed(currPhi[1], base_rps, l, r)
						currPhi[1] = (currPhi[1] + direction * phi_dot_1 * dt) % (2 * np.pi)
						
						x1, y1 = circle_xy(r, currPhi[0], timeSec, base_rps, 0)
						x2, y2 = circle_xy(r, currPhi[1], timeSec, base_rps, 0)
					else:
						angleStep = direction * speed * 2 * pi / refreshRate
						if basicShape == 'diamond':
							perimeter = radii[0] * 4.0
							circum = 2 * pi * radii[0]
							angleStep = angleStep * (perimeter / circum)
						
						currTargetAngle = (currTargetAngle + angleStep) % (2 * pi)
						distractorAngle = (distractorAngle + angleStep) % (2 * pi)
						
						if basicShape == 'circle':
							x1, y1 = circle_xy(radii[0], currTargetAngle, timeSec, speed, 0)
							x2, y2 = circle_xy(radii[0], distractorAngle, timeSec, speed, 0)
						elif basicShape == 'diamond':
							x1, y1 = diamond_xy(radii[0], currTargetAngle)
							x2, y2 = diamond_xy(radii[0], distractorAngle)
						elif basicShape == 'sine_circle':
							x1, y1 = sine_circle_xy(radii[0], currTargetAngle)
							x2, y2 = sine_circle_xy(radii[0], distractorAngle)
						elif basicShape == 'ellipse':
							x1, y1 = ellipse_xy(radii[0], currTargetAngle)
							x2, y2 = ellipse_xy(radii[0], distractorAngle)
						else:
							x1, y1 = circle_xy(radii[0], currTargetAngle)
							x2, y2 = circle_xy(radii[0], distractorAngle)
					
					if next_reversal_idx < len(reversal_times) and timeSec > reversal_times[next_reversal_idx]:
						direction *= -1
						next_reversal_idx += 1
					
					x1 += cx; y1 += cy
					x2 += cx; y2 += cy
					
					fixation.draw()
					if frameN < cueFrames:
						if target_idx == 0:
							blobStim.setFillColor(targetCueColor, log=False)
							blobStim2.setFillColor(distractorCueColor, log=False)
						else:
							blobStim.setFillColor(distractorCueColor, log=False)
							blobStim2.setFillColor(targetCueColor, log=False)
						if target_idx == 0:
							blobStim.setLineColor(targetCueColor, log=False)
							blobStim.setLineWidth(4)
							blobStim.setPos((x1, y1)); blobStim.draw()
							blobStim.setLineColor(None, log=False)
						else:
							blobStim2.setLineColor(targetCueColor, log=False)
							blobStim2.setLineWidth(4)
							blobStim2.setPos((x2, y2)); blobStim2.draw()
							blobStim2.setLineColor(None, log=False)
					else:
						blobStim.setFillColor(identicalBlobColor, log=False)
						blobStim2.setFillColor(identicalBlobColor, log=False)
					blobStim.setLineColor(None, log=False)
					blobStim2.setLineColor(None, log=False)
					blobStim.setPos((x1, y1)); blobStim.draw()
					blobStim2.setPos((x2, y2)); blobStim2.draw()
					
					myWin.flip()
					ts.append(trialClock.getTime() - trial_start_time)
				
				# Response collection
				resp = None
				correct = False
				mouse = event.Mouse(win=myWin)
				clicked = False
				if autoAdvance:
					clicked = True
					correct = True
					fixation.draw()
					blobStim.setFillColor(identicalBlobColor, log=False)
					blobStim2.setFillColor(identicalBlobColor, log=False)
					blobStim.setLineColor(None, log=False)
					blobStim2.setLineColor(None, log=False)
					blobStim.setPos((x1, y1)); blobStim.draw()
					blobStim2.setPos((x2, y2)); blobStim2.draw()
					draw_trajectory(myWin, basicShape, radii[0], cx, cy)
					myWin.flip()
					apply_global_flash(True, blobStim, blobStim2, fixation)
				else:
					t0 = core.getTime()
					while not clicked and core.getTime() - t0 < 10.0:
						fixation.draw()
						blobStim.setFillColor(identicalBlobColor, log=False)
						blobStim2.setFillColor(identicalBlobColor, log=False)
						blobStim.setLineColor(None, log=False)
						blobStim2.setLineColor(None, log=False)
						blobStim.setPos((x1, y1)); blobStim.draw()
						blobStim2.setPos((x2, y2)); blobStim2.draw()
						draw_trajectory(myWin, basicShape, radii[0], cx, cy)
						myWin.flip()
						if mouse.getPressed()[0]:
							mx, my = mouse.getPos()
							d1 = (mx - x1) ** 2 + (my - y1) ** 2
							d2 = (mx - x2) ** 2 + (my - y2) ** 2
							picked = 0 if d1 < d2 else 1
							correct = (picked == target_idx)
							clicked = True
							break
						keys = event.getKeys()
						if 'escape' in keys:
							core.quit()
				
				# Feedback
				if clicked:
					if correct:
						beep_correct.play()
						blobStim.setPos((x1, y1)); blobStim2.setPos((x2, y2))
						apply_global_flash(True, blobStim, blobStim2, fixation)
					else:
						beep_incorrect.play()
						blobStim.setPos((x1, y1)); blobStim2.setPos((x2, y2))
						apply_global_flash(False, blobStim, blobStim2, fixation)
				
				# Timing analysis
				if len(ts) > 1:
					interframe_intervals = np.diff(ts) * 1000.0
					long_frame_indices = np.where(interframe_intervals > longFrameLimit)[0]
				else:
					long_frame_indices = np.array([], dtype=int)
				
				trialnum = len(results)
				timing_blips = int(len(long_frame_indices))
				num_long_frames_after_fixation = int(np.sum(long_frame_indices < timingCheckFrames))
				num_long_frames_after_cue = int(np.sum(long_frame_indices >= timingCheckFrames))
				
				# Write practice trial to TSV
				trialDurTotal = trialClock.getTime() - trial_start_time
				reversal_str = '\t'.join([str(round(r, 4)) for r in reversal_times])
				if len(reversal_times) < 10:
					reversal_str += '\t' + '\t'.join(['-999'] * (10 - len(reversal_times)))
				
				print(trialnum, subject, session, part_name, cond, basicShape, 2, speed, motion_rule, 'practice',
					  0, 'n/a', '-999', -999, -999,
					  round(initialAngle, 4), round(initialOtherAngle, 4),
					  cueFrames, timingCheckFrames, int(correct), round(trialDurTotal, 3), 1, target_idx,
					  len(reversal_times), timing_blips, num_long_frames_after_fixation, num_long_frames_after_cue,
					  sep='\t', end='\t', file=dataFile)
				print(reversal_str, file=dataFile)
				dataFile.flush()
				results.append({'part': part_name, 'condition': cond, 'motionRule': motion_rule, 'trialKind': 'practice', 'speed': speed, 'stairValue': -999, 'staircaseIndex': 0, 'staircaseWithinCondition': 'n/a', 'stairKey': '-999', 'correct': bool(correct)})
		
		# Show break before main trials
		if practice_trials:
			show_break_screen(f"Practice trials complete. Press SPACE to begin the actual {part_name} trials.")
		
		# Set up staircase visualization
		if stairViz is not None:
			panel_title = f'{part_name} staircase tracks'
			panel_subtitle = 'Two interleaved staircases per condition (higher rps at top)'
			stairViz.set_layout(part_name, part_specs, title=panel_title, subtitle=panel_subtitle, y_bounds=(0.4, 2.4))
			# counters for per-condition trial numbers
			cond_counters = {spec['stairKey']: {1: 0, 2: 0, 'attention': 0} for spec in part_specs}
		
		# Track how many trials have been completed within this part so the 100-trial
		# break schedule resets at the start of each part.
		start_index = len(results)
		for ti, thisTrial in enumerate(trial_list):
			# insert scheduled break every 100 completed trials WITHIN this part
			completed_in_part = len(results) - start_index
			if completed_in_part > 0 and completed_in_part % 100 == 0:
				show_break_screen(f"Please take a short break. {completed_in_part} trials completed in {part_name}. Press SPACE to continue.")
			# determine center
			cond = thisTrial['condition']
			condition_key = thisTrial.get('conditionKey', thisTrial.get('stairKey', '-999'))
			motion_rule = thisTrial.get('motionRule', 'standard')
			trial_kind = thisTrial.get('trialKind', 'staircase')
			staircase_index = thisTrial.get('staircaseIndex', 0)
			staircase_label = f'stair{staircase_index}' if staircase_index in (1, 2) else 'n/a'
			stair_key = thisTrial.get('stairKey', '-999')
			stair_start_speed = thisTrial.get('stairStartSpeed', -999)
			if part_name == 'Part1':
				if cond == 'centred':
					cx = trajectory_center_x_deg[0]
				elif cond == 'near_displaced':
					cx = trajectory_center_x_deg[1]
				elif cond == 'far_displaced':
					cx = trajectory_center_x_deg[2]
				cy = 0.0
				basicShape = 'circle'
			else:  # Part2 shapes
				cx = 0.0
				cy = 0.0
				basicShape = cond  # 'diamond','sine_circle','ellipse'

			if trial_kind == 'attention_check':
				speed = attention_check_speed
				stair_value = -999
			else:
				stair_value = speed_staircases[thisTrial['stairKey']].next()
				speed = stair_value_to_speed(stair_value)

			# starting angles (two objects opposite)
			currTargetAngle = random.random() * 2 * pi
			distractorAngle = (currTargetAngle + pi) % (2 * pi)
			initialAngle = currTargetAngle
			initialOtherAngle = distractorAngle
			
			currPhi = [random.random() * 2 * pi, 0.0]
			currPhi[1] = (currPhi[0] + pi) % (2 * pi)
			direction = 1
			reversal_times = get_reversal_times(trial_duration_sec)
			next_reversal_idx = 0

			# cue which is target: target is object 0
			target_idx = 0

			# Record trial start time
			trial_start_time = trialClock.getTime()
			ts = []

			# frame loop
			for frameN in range(trialDurFrames):
				timeSec = frameN / refreshRate
				if part_name == 'Part1' and basicShape == 'circle' and motion_rule == 'varying':
					# Part 2: use the same eccentricity-based rule for centred and displaced conditions.
					# When l=0 this reduces to fixed-speed circular motion.
					dt = 1.0 / refreshRate
					l = np.sqrt(cx * cx + cy * cy)
					base_rps = speed
					r = radii[0]

					# Precompute trial-level debug values on first frame.
					if frameN == 0:
						phi_samples = np.linspace(0, 2 * np.pi, 360, endpoint=False)
						trial_rhos = rho_from_phi(phi_samples, l, r)
						trial_phi_dots = phi_dot_log_eccentricity_base_speed(phi_samples, base_rps, l, r)
						print(
							f"DEBUG: Part 2 varying-speed trial {ti+1} | condition={cond} | "
							f"base_rps={base_rps:.3f} | l={l:.3f} | r={r:.3f} | "
							f"min_rho={np.min(trial_rhos):.6f} | max_rho={np.max(trial_rhos):.6f} | "
							f"min_phi_dot={np.min(trial_phi_dots):.6f} | max_phi_dot={np.max(trial_phi_dots):.6f}"
						)

					# Update object 1 (target) independently
					phi_dot_0 = phi_dot_log_eccentricity_base_speed(currPhi[0], base_rps, l, r)
					# Calculate angle (phi) it should be at after moving for dt seconds, applying direction for reversals
					currPhi[0] = (currPhi[0] + direction * phi_dot_0 * dt) % (2 * np.pi)

					# Update object 2 (distractor) independently
					phi_dot_1 = phi_dot_log_eccentricity_base_speed(currPhi[1], base_rps, l, r)
					currPhi[1] = (currPhi[1] + direction * phi_dot_1 * dt) % (2 * np.pi)

					x1, y1 = circle_xy(r, currPhi[0], timeSec, base_rps, 0)
					x2, y2 = circle_xy(r, currPhi[1], timeSec, base_rps, 0)
				else:
					# Parts 1 & 3: Both objects move together at the same angular speed
					angleStep = direction * speed * 2 * pi / refreshRate
					# adjust speed for diamond as in practice
					if basicShape == 'diamond':
						perimeter = radii[0] * 4.0
						circum = 2 * pi * radii[0]
						angleStep = angleStep * (perimeter / circum)

					currTargetAngle = (currTargetAngle + angleStep) % (2 * pi)
					#Calculate angle of the distractor (distractorAngle) which is always pi radians away from the target, then apply same angleStep for reversals
					distractorAngle = (distractorAngle + angleStep) % (2 * pi)

					# compute positions depending on shape
					if basicShape == 'circle':
						x1, y1 = circle_xy(radii[0], currTargetAngle, timeSec, speed, 0)
						x2, y2 = circle_xy(radii[0], distractorAngle, timeSec, speed, 0)
					elif basicShape == 'diamond':
						x1, y1 = diamond_xy(radii[0], currTargetAngle)
						x2, y2 = diamond_xy(radii[0], distractorAngle)
					elif basicShape == 'sine_circle':
						x1, y1 = sine_circle_xy(radii[0], currTargetAngle)
						x2, y2 = sine_circle_xy(radii[0], distractorAngle)
					elif basicShape == 'ellipse':
						x1, y1 = ellipse_xy(radii[0], currTargetAngle)
						x2, y2 = ellipse_xy(radii[0], distractorAngle)
					else:
						x1, y1 = circle_xy(radii[0], currTargetAngle)
						x2, y2 = circle_xy(radii[0], distractorAngle)

				if next_reversal_idx < len(reversal_times) and timeSec > reversal_times[next_reversal_idx]:
					direction *= -1
					next_reversal_idx += 1

				# offset from fixation
				x1 += cx; y1 += cy
				x2 += cx; y2 += cy

				# draw everything for this frame
				fixation.draw()
				# During the cue interval show the target as white and distractor as red
				if frameN < cueFrames:
					if target_idx == 0:
						blobStim.setFillColor(targetCueColor, log=False)
						blobStim2.setFillColor(distractorCueColor, log=False)
					else:
						blobStim.setFillColor(distractorCueColor, log=False)
						blobStim2.setFillColor(targetCueColor, log=False)
					# also draw outline cue on the target for extra salience
					if target_idx == 0:
						blobStim.setLineColor(targetCueColor, log=False)
						blobStim.setLineWidth(4)
						blobStim.setPos((x1, y1)); blobStim.draw()
						blobStim.setLineColor(None, log=False)
					else:
						blobStim2.setLineColor(targetCueColor, log=False)
						blobStim2.setLineWidth(4)
						blobStim2.setPos((x2, y2)); blobStim2.draw()
						blobStim2.setLineColor(None, log=False)
				else:
					# after cue interval both objects use the identical blob color for tracking
					blobStim.setFillColor(identicalBlobColor, log=False)
					blobStim2.setFillColor(identicalBlobColor, log=False)
				blobStim.setLineColor(None, log=False)
				blobStim2.setLineColor(None, log=False)
				blobStim.setPos((x1, y1)); blobStim.draw()
				blobStim2.setPos((x2, y2)); blobStim2.draw()

				myWin.flip()
				ts.append(trialClock.getTime() - trial_start_time)

			# response collection: ask participant to click the target
			# draw the objects continuously while waiting so they remain visible
			resp = None
			correct = False
			mouse = event.Mouse(win=myWin)
			clicked = False
			if autoAdvance:
				clicked = True
				correct = True
				fixation.draw()
				blobStim.setFillColor(identicalBlobColor, log=False)
				blobStim2.setFillColor(identicalBlobColor, log=False)
				blobStim.setLineColor(None, log=False)
				blobStim2.setLineColor(None, log=False)
				blobStim.setPos((x1, y1)); blobStim.draw()
				blobStim2.setPos((x2, y2)); blobStim2.draw()
				draw_trajectory(myWin, basicShape, radii[0], cx, cy)
				myWin.flip()
				# provide visual feedback for auto-advance trials as if correct
				apply_global_flash(True, blobStim, blobStim2, fixation)
			else:
				t0 = core.getTime()
				while not clicked and core.getTime() - t0 < 10.0:
					fixation.draw()
					blobStim.setFillColor(identicalBlobColor, log=False)
					blobStim2.setFillColor(identicalBlobColor, log=False)
					blobStim.setLineColor(None, log=False)
					blobStim2.setLineColor(None, log=False)
					blobStim.setPos((x1, y1)); blobStim.draw()
					blobStim2.setPos((x2, y2)); blobStim2.draw()
					draw_trajectory(myWin, basicShape, radii[0], cx, cy)
					myWin.flip()
					if mouse.getPressed()[0]:
						mx, my = mouse.getPos()
						# determine which object was closer
						d1 = (mx - x1) ** 2 + (my - y1) ** 2
						d2 = (mx - x2) ** 2 + (my - y2) ** 2
						picked = 0 if d1 < d2 else 1
						correct = (picked == target_idx)
						clicked = True
						break
					keys = event.getKeys()
					if 'escape' in keys:
						core.quit()

			# feedback: play sound and visual flash
			if clicked:
				if correct:
					beep_correct.play()
					# Flash both objects green twice
					# ensure positions are set
					blobStim.setPos((x1, y1)); blobStim2.setPos((x2, y2))
					apply_global_flash(True, blobStim, blobStim2, fixation)
				else:
					beep_incorrect.play()
					# Flash both objects bright orange twice
					blobStim.setPos((x1, y1)); blobStim2.setPos((x2, y2))
					apply_global_flash(False, blobStim, blobStim2, fixation)

			if len(ts) > 1:
				interframe_intervals = np.diff(ts) * 1000.0
				long_frame_indices = np.where(interframe_intervals > longFrameLimit)[0]
			else:
				long_frame_indices = np.array([], dtype=int)
			trialnum = len(results)
			timing_blips = int(len(long_frame_indices))
			num_long_frames_after_fixation = int(np.sum(long_frame_indices < timingCheckFrames))
			num_long_frames_after_cue = int(np.sum(long_frame_indices >= timingCheckFrames))
			if timing_blips > 0 or os.environ.get('MOT_VIS_DEBUG', '0').strip().lower() in ('1', 'true', 'yes', 'y', 'on'):
				print(
					f"TIMING_CHECK: trialnum={trialnum} blips={timing_blips} "
					f"after_fixation={num_long_frames_after_fixation} after_cue={num_long_frames_after_cue} "
					f"longFrameLimit={longFrameLimit}"
				)

			if stairViz is not None and trial_kind == 'attention_check':
				cond_counters[condition_key]['attention'] += 1
				attention_trial_num = cond_counters[condition_key]['attention']
				stairViz.add_point(condition_key, 0, speed, trial_num=attention_trial_num, kind='attention')

			if trial_kind != 'attention_check':
				# PsychoPy `StairHandler` semantics: pass `True` for a correct
				# response and `False` for incorrect. Use the raw `correct`
				# boolean so step directions follow `nUp`/`nDown` as configured.
				speed_staircases[thisTrial['stairKey']].addResponse(bool(correct))
				if stairViz is not None:
					# increment per-condition/staircase counter and plot at that trial index
					cond_counters[condition_key][staircase_index] += 1
					trial_num = cond_counters[condition_key][staircase_index]
					stairViz.add_point(condition_key, staircase_index, speed, trial_num=trial_num)

			# Write comprehensive trial result to data file
			trialDurTotal = trialClock.getTime() - trial_start_time
			
			# Format reversal times for output
			reversal_str = '\t'.join([str(round(r, 4)) for r in reversal_times])
			if len(reversal_times) < 10:
				reversal_str += '\t' + '\t'.join(['-999'] * (10 - len(reversal_times)))
			
			# Write all trial data
			print(trialnum, subject, session, part_name, cond, basicShape, 2, speed, motion_rule, trial_kind,
				  staircase_index, staircase_label, stair_key, stair_start_speed, stair_value,
				  round(initialAngle, 4), round(initialOtherAngle, 4),
				  cueFrames, timingCheckFrames, int(correct), round(trialDurTotal, 3), 1, target_idx,
				  len(reversal_times), timing_blips, num_long_frames_after_fixation, num_long_frames_after_cue,
				  sep='\t', end='\t', file=dataFile)
			print(reversal_str, file=dataFile)
			dataFile.flush()
			results.append({'part': part_name, 'condition': cond, 'motionRule': motion_rule, 'trialKind': trial_kind, 'speed': speed, 'stairValue': stair_value, 'staircaseIndex': staircase_index, 'staircaseWithinCondition': staircase_label, 'stairKey': stair_key, 'correct': bool(correct)})

	if 'part1' in parts_to_run:
		run_part('Part1', trials['part1'])
		# provide a break after Part 1 completes
		show_break_screen('Part 1 complete. Please take a longer break if needed. Press SPACE to continue to Part 2.')
	if 'part2' in parts_to_run:
		run_part('Part2', trials['part2'])

	# Final summary
	n_correct = sum(1 for r in results if r['correct'])
	print('\nExperiment complete. Total correct:', n_correct, 'out of', len(results))
	print('Results saved to:', datafileName+'.tsv')
	dataFile.close()
	myWin.close()
	# Visualiser debug summary per panel
	try:
		if 'stairViz' in globals() and stairViz is not None and os.environ.get('MOT_VIS_DEBUG', '0').strip() in ('1', 'true', 'yes'):
			for cond_key, slots in stairViz.panel_data.items():
				counts = {1: sum(1 for s in slots[1] if s is not None), 2: sum(1 for s in slots[2] if s is not None)}
				attention_count = sum(1 for s in stairViz.attention_data.get(cond_key, []) if s is not None)
				print(f"VIS_DEBUG: panel {cond_key} counts: {counts}, attention={attention_count}")
	except Exception:
		pass
