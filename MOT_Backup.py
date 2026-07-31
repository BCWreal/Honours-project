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
from psychopy import visual, core, event, sound, monitors, data, gui, info, logging
import numpy as np, random, time, os
from collections import deque
from math import pi, cos, sin
from pathlib import Path
import atexit

# Eye-tracking integration based on the Exp1b session wrapper.
eyetracking = True

HAS_PYLINK = False
_PYLINK_IMPORT_ERROR = None
try:
	import pylink
	from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy
	HAS_PYLINK = True
except Exception as exc:
	pylink = None
	EyeLinkCoreGraphicsPsychoPy = None
	_PYLINK_IMPORT_ERROR = exc


def _default_eye_mode():
	if HAS_PYLINK and eyetracking:
		return 'live'
	return 'off'


EYE = {
	'mode': os.environ.get('MOT_EYE_MODE', _default_eye_mode()).strip().lower(),
	'host_ip': os.environ.get('MOT_EYE_HOST_IP', '100.1.1.1'),
	'calibration_type': os.environ.get('MOT_EYE_CALIBRATION_TYPE', 'HV9'),
	'log_edf': os.environ.get('MOT_EYE_LOG_EDF', '1').strip().lower() in ('1', 'true', 'yes', 'y', 'on'),
	'fixation_window_deg': float(os.environ.get('MOT_EYE_FIXATION_WINDOW_DEG', '1.5')),
	'fix_break_consec_frames': int(os.environ.get('MOT_EYE_FIX_BREAK_CONSEC_FRAMES', '1')),
}
if not HAS_PYLINK:
	EYE['mode'] = 'off'
if EYE['mode'] not in ('off', 'dummy', 'live'):
	EYE['mode'] = 'off'


class EyeLinkSession:
	"""Wrapper around pylink for one session of the MOT experiment."""

	def __init__(self, mode, edf_basename, host_ip, calibration_type='HV9', log_edf=True):
		self.mode = mode
		self.edf_basename = edf_basename[:8]
		self.edf_filename = self.edf_basename + '.EDF'
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
		self.fix_win_deg = 1.5
		self.eye_used = 0
		self.eyelink_ver = 0

	def is_active(self):
		return self.mode in ('dummy', 'live') and self.tracker is not None

	def open(self):
		if self.mode == 'off':
			return
		if not HAS_PYLINK:
			raise RuntimeError(
				"EyeLink mode '{}' requested but pylink could not be imported (error: {}). "
				"Install pylink or set MOT_EYE_MODE=off."
				.format(self.mode, _PYLINK_IMPORT_ERROR)
			)
		if self.mode == 'dummy':
			self.tracker = pylink.EyeLink(None)
		else:
			self.tracker = pylink.EyeLink(self.host_ip)
		self.tracker.openDataFile(self.edf_filename)
		if self.log_edf and self.is_active():
			self.tracker.sendCommand("add_file_preamble_text 'MOT polar-angle tracking'")
		self.tracker.setOfflineMode()
		if self.mode == 'live':
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
			self.tracker.sendCommand('file_event_filter = ' + file_event_flags)
			self.tracker.sendCommand('file_sample_data  = ' + file_sample_flags)
		self.tracker.sendCommand('link_event_filter = ' + link_event_flags)
		self.tracker.sendCommand('link_sample_data  = ' + link_sample_flags)
		self.tracker.sendCommand('calibration_type = ' + self.calibration_type)

	def setup_graphics(self, win, fix_window_deg=1.5):
		if not self.is_active():
			return
		self.scn_w, self.scn_h = win.size
		self.fix_win_deg = float(fix_window_deg)
		self.tracker.sendCommand('screen_pixel_coords = 0 0 {:d} {:d}'.format(self.scn_w - 1, self.scn_h - 1))
		self.tracker.sendMessage('DISPLAY_COORDS 0 0 {:d} {:d}'.format(self.scn_w - 1, self.scn_h - 1))
		if EyeLinkCoreGraphicsPsychoPy is None:
			return
		self.genv = EyeLinkCoreGraphicsPsychoPy(self.tracker, win)
		self.genv.setCalibrationColors((-1, -1, -1), win.color)
		self.genv.setTargetType('circle')
		self.genv.setTargetSize(24)
		self.genv.setCalibrationSounds('', '', '')
		pylink.openGraphicsEx(self.genv)
		px_per_cm = self.scn_w / float(DEFAULT_MONITOR_WIDTH_CM)
		cm_per_deg = DEFAULT_MONITOR_DISTANCE_CM * np.tan(np.deg2rad(1.0))
		self.fix_win_px = self.fix_win_deg * (px_per_cm * cm_per_deg)
		self.fix_el_x = self.scn_w / 2.0
		self.fix_el_y = self.scn_h / 2.0

	def calibrate(self):
		if not self.is_active():
			return
		try:
			self.tracker.doTrackerSetup()
		except RuntimeError as e:
			print('EyeLink calibration error:', e)
			try:
				self.tracker.exitCalibration()
			except Exception:
				pass

	def drift_correct(self):
		if not self.is_active():
			return
		try:
			self.tracker.setOfflineMode()
			self.tracker.sendCommand('clear_screen 0')
			err = self.tracker.doDriftCorrect(int(self.fix_el_x), int(self.fix_el_y), 1, 1)
			if err == pylink.ESC_KEY:
				pass
		except RuntimeError as e:
			print('Drift correct error:', e)

	def start_trial(self, trial_id):
		if not self.is_active():
			return
		self.tracker.setOfflineMode()
		self.tracker.sendCommand('clear_screen 0')
		self.tracker.sendMessage('TRIALID {:d}'.format(trial_id))
		self.tracker.sendCommand("record_status_message 'Trial {:d}'".format(trial_id))
		try:
			file_rec = 1 if self.log_edf else 0
			self.tracker.startRecording(file_rec, file_rec, 1, 1)
		except RuntimeError as e:
			print('startRecording error:', e)
			return
		pylink.pumpDelay(50)
		eu = self.tracker.eyeAvailable()
		if eu == 2:
			eu = 0
		self.eye_used = eu
		if eu == 1:
			self.tracker.sendMessage('EYE_USED 1 RIGHT')
		else:
			self.tracker.sendMessage('EYE_USED 0 LEFT')
		self.tracker.sendMessage('TRIAL_START {:d}'.format(trial_id))

	def stop_trial(self, trial_id, result_code=0):
		if not self.is_active():
			return
		if self.tracker.isConnected():
			try:
				self.tracker.sendMessage('TRIAL_END {:d}'.format(trial_id))
				pylink.pumpDelay(50)
				self.tracker.stopRecording()
				self.tracker.sendMessage('TRIAL_RESULT {:d}'.format(result_code))
			except Exception as e:
				print('stop_trial error:', e)

	def send_message(self, msg):
		if not self.is_active():
			return
		try:
			self.tracker.sendMessage(msg)
		except Exception:
			pass

	def get_gaze(self, prev_sample):
		"""Return (in_window, sample, distance_deg) for the latest EyeLink gaze sample."""
		if not self.is_active():
			return True, prev_sample, 0.0
		try:
			new_sample = self.tracker.getNewestSample()
		except Exception:
			return False, prev_sample, np.inf
		if new_sample is None:
			return False, prev_sample, np.inf
		if prev_sample is not None and new_sample.getTime() == prev_sample.getTime():
			return None, prev_sample, None
		if self.eye_used == 1 and new_sample.isRightSample():
			gx, gy = new_sample.getRightEye().getGaze()
		elif self.eye_used == 0 and new_sample.isLeftSample():
			gx, gy = new_sample.getLeftEye().getGaze()
		else:
			return False, new_sample, np.inf
		miss = -32768
		if gx == miss or gy == miss:
			return False, new_sample, np.inf
		dist_px = float(np.hypot(gx - self.fix_el_x, gy - self.fix_el_y))
		px_per_deg = self.fix_win_px / self.fix_win_deg if self.fix_win_deg else np.inf
		dist_deg = dist_px / px_per_deg if px_per_deg else np.inf
		return (dist_deg <= self.fix_win_deg), new_sample, dist_deg

	def close(self, output_dir='', base_name=''):
		if not self.is_active():
			return
		if self.tracker.isConnected():
			try:
				if self.tracker.isRecording() == pylink.TRIAL_OK:
					pylink.pumpDelay(100)
					self.tracker.stopRecording()
				self.tracker.setOfflineMode()
				self.tracker.sendCommand('clear_screen 0')
				pylink.msecDelay(500)
				try:
					self.tracker.closeDataFile()
				except Exception as e:
					print('EDF close error:', e)
				if self.log_edf and output_dir and base_name:
					local_edf = os.path.join(output_dir, base_name + '.EDF')
					try:
						self.tracker.receiveDataFile(self.edf_filename, local_edf)
						print('EDF saved to:', local_edf)
					except RuntimeError as e:
						print('EDF transfer error:', e)
				self.tracker.close()
			except Exception as e:
				print('EyeLink close error:', e)

	# Compatibility aliases used by the existing MOT code path.
	def start_recording(self, trial_num, calib_trial=True, widthPix=None, heightPix=None):
		self.start_trial(trial_num)

	def stop_recording(self, trial_num=None, result_code=0):
		if trial_num is None:
			trial_num = -1
		self.stop_trial(trial_num, result_code=result_code)


tracker = None


def draw_center_fixation(win, fixation, fixation_blank, fixation_point, frame_index=None):
	"""Draw the centered fixation used in AH Circular-style trials."""
	if frame_index is None:
		fixation.draw()
	else:
		if frame_index % 2:
			fixation.draw()
		else:
			fixation_blank.draw()
	fixation_point.draw()


# Ensure data directory exists to match earlier messages and avoid warnings
data_dir = Path('dataRaw')
if not data_dir.exists():
	try:
		data_dir.mkdir(parents=True, exist_ok=True)
	except Exception:
		pass

def get_screen_info(prefer_second=False):
	"""Return (screen_index, [width, height]) for the requested display.

	prefer_second=False uses the primary screen for the experiment.
	prefer_second=True uses the second screen for the visualiser if available,
	otherwise it falls back to the primary screen.
	"""
	try:
		import pyglet
		display = pyglet.canvas.get_display()
		screens = display.get_screens()

		if prefer_second and len(screens) > 1:
			screen_index = 1
			screen = screens[1]
		else:
			screen_index = 0
			screen = screens[0]

		return screen_index, [int(screen.width), int(screen.height)]
	except Exception:
		return 0, [1920, 1080]


def get_screen_size_by_index(screen_index, fallback_prefer_second=False):
	"""Return [width, height] for a specific monitor index.

	This is used separately for the experiment monitor and the visualiser
	monitor, because they may have different pixel resolutions.
	"""
	try:
		import pyglet
		display = pyglet.canvas.get_display()
		screens = display.get_screens()
		idx = int(screen_index)
		if 0 <= idx < len(screens):
			screen = screens[idx]
			return [int(screen.width), int(screen.height)]
	except Exception:
		pass
	_, fallback_size = get_screen_info(prefer_second=fallback_prefer_second)
	return fallback_size


# Ensure a default monitor is present and has all metadata needed for deg units.
# PsychoPy needs BOTH physical width and viewing distance when units='deg'.
DEFAULT_MONITOR_NAME = 'default_monitor'
DEFAULT_MONITOR_WIDTH_CM = 52.0
DEFAULT_MONITOR_DISTANCE_CM = 57.0


def configure_default_monitor(size_pix=None):
	"""Create/update default_monitor so PsychoPy can convert deg to pixels."""
	try:
		if size_pix is None:
			_, size_pix = get_screen_info(prefer_second=False)
		m = monitors.Monitor(DEFAULT_MONITOR_NAME)
		m.setSizePix(tuple(size_pix))
		m.setWidth(DEFAULT_MONITOR_WIDTH_CM)
		m.setDistance(DEFAULT_MONITOR_DISTANCE_CM)
		try:
			m.save()
		except Exception:
			pass
		return m
	except Exception as e:
		print('Warning: could not configure default monitor:', e)
		return monitors.Monitor(DEFAULT_MONITOR_NAME, width=DEFAULT_MONITOR_WIDTH_CM, distance=DEFAULT_MONITOR_DISTANCE_CM)


configure_default_monitor()

def _as_bool(value):
	"""Coerce PsychoPy dialog values into a real bool."""
	if isinstance(value, str):
		return value.strip().lower() in ('1', 'true', 'yes', 'y', 'on')
	return bool(value)


def run_pre_experiment_screen_check(default_refresh_rate=100.0):
	"""
	Show a pre-experiment dialog for screen choice, fullscreen mode, and refresh rate.
	Then open a temporary window to confirm the actual detected resolution and,
	optionally, run PsychoPy's refresh timing check.
	"""
	# Best-effort screen discovery for sensible defaults.
	try:
		import pyglet
		display = pyglet.canvas.get_display()
		screens = display.get_screens()
		n_screens = len(screens)
	except Exception:
		screens = []
		n_screens = 1

	default_exp_screen = 0
	default_viz_screen = 1 if n_screens > 1 else 0

	settings = {
		'Screen to use': default_exp_screen,
		'Visualizer screen': default_viz_screen,
		'Fullscreen': True,
		'Screen refresh rate': float(default_refresh_rate),
		'Log EDF File': EYE['log_edf'],
		'Check refresh etc': True,
	}

	dlg = gui.DlgFromDict(
		dictionary=settings,
		title='MOT screen setup',
		order=['Screen to use', 'Visualizer screen', 'Fullscreen', 'Screen refresh rate', 'Log EDF File', 'Check refresh etc'],
		tip={
			'Screen to use': '0 means primary screen, 1 means second screen.',
			'Visualizer screen': 'Usually 1 if a second monitor is attached; use 0 if only one screen is available.',
			'Fullscreen': 'Recommended for accurate timing.',
			'Screen refresh rate': 'Enter the intended monitor refresh rate in Hz.',
			'Log EDF File': 'If checked, the experimenter logs and transfers an EDF file at shutdown.',
			'Check refresh etc': 'Runs PsychoPy RuntimeInfo to estimate actual refresh timing before the experiment.',
		},
	)
	if not dlg.OK:
		print('User cancelled from screen setup dialog.')
		core.quit()

	try:
		exp_screen_index = int(settings['Screen to use'])
	except Exception:
		exp_screen_index = 0
	try:
		viz_screen_index = int(settings['Visualizer screen'])
	except Exception:
		viz_screen_index = 1 if n_screens > 1 else 0

	fullscr_requested = _as_bool(settings['Fullscreen'])
	check_refresh = _as_bool(settings['Check refresh etc'])
	EYE['log_edf'] = _as_bool(settings['Log EDF File'])
	try:
		chosen_refresh = float(settings['Screen refresh rate'])
	except Exception:
		chosen_refresh = float(default_refresh_rate)

	# Requested size comes from detected screen size. In fullscreen mode, PsychoPy
	# may still return a slightly different actual size, so we read temp_win.size below.
	_, fallback_size = get_screen_info(prefer_second=False)
	try:
		if screens and 0 <= exp_screen_index < len(screens):
			exp_win_size = [int(screens[exp_screen_index].width), int(screens[exp_screen_index].height)]
		else:
			exp_win_size = fallback_size
	except Exception:
		exp_win_size = fallback_size

	# Measure the visualiser monitor separately. This matters when the second
	# monitor has a different resolution from the experiment monitor.
	viz_win_size = get_screen_size_by_index(viz_screen_index, fallback_prefer_second=True)

	refresh_msg1 = 'Refresh rate was not checked.'
	refresh_msg2 = ''
	refreshRateWrong = False
	actual_size = exp_win_size
	measured_fps = None
	temp_win = None

	try:
		temp_win = visual.Window(
			size=exp_win_size,
			screen=exp_screen_index,
			units='deg',
			fullscr=fullscr_requested,
			color=(0, 0, 0),
			monitor=configure_default_monitor(exp_win_size),
			allowGUI=False,
			checkTiming=False,
		)
		actual_size = [int(temp_win.size[0]), int(temp_win.size[1])]

		if check_refresh:
			runInfo = info.RunTimeInfo(
				win=temp_win,
				refreshTest='grating',
				verbose=False,
				userProcsDetailed=False,
			)
			median_ms = runInfo.get('windowRefreshTimeMedian_ms', None)
			if median_ms:
				median_fps = 1000.0 / median_ms
				measured_fps = median_fps
				refresh_msg1 = 'Median frames per second = ' + str(round(median_fps, 1))
				refresh_tolerance_pct = 3
				pct_off = abs((median_fps - chosen_refresh) / chosen_refresh) if chosen_refresh else 0
				refreshRateWrong = pct_off > (refresh_tolerance_pct / 100.0)
				if refreshRateWrong:
					refresh_msg1 += ' BUT program assumes ' + str(round(chosen_refresh, 1))
					refresh_msg2 = 'which is off by more than ' + str(refresh_tolerance_pct) + '%'
				else:
					refresh_msg1 += ', close enough to desired ' + str(round(chosen_refresh, 1))
			else:
				refresh_msg1 = 'Refresh check ran, but median refresh time was unavailable.'
	except Exception as e:
		refresh_msg1 = 'Screen check warning: ' + str(e)
	finally:
		try:
			if temp_win is not None:
				temp_win.close()
		except Exception:
			pass

	# Show a confirmation/warning dialog before the real experiment window opens.
	# The user can accept the measured refresh rate or override it manually.
	refresh_to_use_default = measured_fps if measured_fps is not None else chosen_refresh
	confirm = gui.Dlg(title='MOT screen check result')
	confirm.addText(refresh_msg1, color='Red' if refreshRateWrong else 'Black')
	if refresh_msg2:
		confirm.addText(refresh_msg2, color='Red')
	if measured_fps is not None:
		confirm.addText('Measured refresh rate: ' + str(round(measured_fps, 2)) + ' Hz', color='Black')
	else:
		confirm.addText('Measured refresh rate unavailable; using manually entered value unless changed below.', color='GoldenRod')
	confirm.addField('Refresh rate to use for experiment (Hz):', round(float(refresh_to_use_default), 3))
	if actual_size != exp_win_size:
		confirm.addText(
			'Requested resolution ' + str(exp_win_size[0]) + 'x' + str(exp_win_size[1]) +
			' but actual window is ' + str(actual_size[0]) + 'x' + str(actual_size[1]) +
			'. The experiment will use the actual size.',
			color='GoldenRod'
		)
	else:
		confirm.addText('Detected experiment window: ' + str(actual_size[0]) + ' x ' + str(actual_size[1]), color='Black')
	confirm.addText('Experiment screen: ' + str(exp_screen_index) + ' | Visualizer screen: ' + str(viz_screen_index), color='DimGrey')
	confirm.addText('Detected visualizer window: ' + str(viz_win_size[0]) + ' x ' + str(viz_win_size[1]), color='Black')
	confirm.addText('Press OK to continue, or Cancel to quit.', color='DimGrey')
	confirm.show()
	if not confirm.OK:
		print('User cancelled after screen check.')
		core.quit()
	try:
		chosen_refresh = float(confirm.data[0])
	except Exception:
		chosen_refresh = float(refresh_to_use_default)

	return {
		'exp_screen_index': exp_screen_index,
		'viz_screen_index': viz_screen_index,
		'exp_win_size': actual_size,
		'viz_win_size': viz_win_size,
		'fullscr': fullscr_requested,
		'refreshRate': chosen_refresh,
		'refreshRateWrong': refreshRateWrong,
		'refreshMsg1': refresh_msg1,
		'refreshMsg2': refresh_msg2,
	}



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
	"""Convert PsychoPy internal staircase value to displayed rps.

	The internal staircase value is inverted so that PsychoPy's 3-down step
	decreases the internal value but increases the displayed rps.
	"""
	return -stair_value


def display_speed_to_stair_value(display_speed):
	"""Convert displayed rps into PsychoPy's internal staircase value."""
	return -display_speed


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
	"""Build practice trials with part-specific speeds.

	Part 1: two practice trials per condition at 0.5 and 1.0 rps.
	Part 2: three practice trials per condition at 0.5, 0.7, and 1.0 rps.
	"""
	trials = []

	if part_name == 'Part1':
		practice_speeds = [0.5, 1.0]
	elif part_name == 'Part2':
		practice_speeds = [0.5, 0.7, 1.0]
	else:
		practice_speeds = [1.0]

	for spec in trial_specs:
		for practice_speed in practice_speeds:
			trials.append({
				'part': part_name,
				'condition': spec['condition'],
				'motionRule': spec.get('motionRule', 'standard'),
				'conditionKey': spec['stairKey'],
				'trialKind': 'practice',
				'speed': practice_speed,
			})
	if part_name == 'Part2':
		ellipse_trials = [trial for trial in trials if trial['condition'] == 'ellipse']
		rotations = build_part2_ellipse_rotations(len(ellipse_trials))
		for trial, rotation_rad in zip(ellipse_trials, rotations):
			trial['ellipseRotationRad'] = rotation_rad
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
		if part_name == 'Part2' and spec['condition'] == 'ellipse':
			ellipse_trials = list(state['queues'][1]) + list(state['queues'][2])
			rotations = build_part2_ellipse_rotations(len(ellipse_trials))
			for trial, rotation_rad in zip(ellipse_trials, rotations):
				trial['ellipseRotationRad'] = rotation_rad
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


def create_staircase_window(screen_index=None, fullscr=True, win_size=None):
	try:
		if screen_index is None:
			screen_index, detected_size = get_screen_info(prefer_second=True)
		else:
			detected_size = get_screen_size_by_index(screen_index, fallback_prefer_second=True)

		if win_size is None:
			win_size = detected_size

		print(f"Staircase window: target_screen={screen_index}, detected_size={detected_size}, using_size={win_size}")

		# HiDPI/Retina displays can report a framebuffer size that is larger than
		# the visible client area. For the pixel-based visualiser, that makes the
		# graph look magnified and clipped. Prefer non-Retina scaling when the
		# PsychoPy version supports it; fall back safely if not.
		window_kwargs = dict(
			size=win_size,
			screen=screen_index,
			fullscr=fullscr,
			winType='pyglet',
			units='pix',
			color=[-0.9, -0.9, -0.9],
			allowGUI=False,
			waitBlanking=False,
			autoLog=False,
			checkTiming=False,
		)
		try:
			win = visual.Window(**window_kwargs, useRetina=False)
		except TypeError:
			win = visual.Window(**window_kwargs)

		print("Actual visualizer PsychoPy win.size:", win.size)
		try:
			print("Actual visualizer client size:", win.winHandle.get_size())
		except Exception:
			pass
		return win
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


def show_text_screen(win, lines, key_list=None, height=0.8, wrap_width=40):
	"""Show a centered multi-line text screen and wait for a response key."""
	if key_list is None:
		key_list = ['space', 'spacebar', 'escape']
	activate_window(win)
	event.clearEvents(eventType='keyboard')
	text = visual.TextStim(
		win,
		text='\n'.join(lines),
		pos=(0, 0),
		height=26,
		color='white',
		alignText='center',
		wrapWidth=1200,
	)
	text.draw()
	win.flip()
	while True:
		keys = event.waitKeys(keyList=key_list)
		if not keys:
			continue
		if 'escape' in keys:
			core.quit()
		return keys[0]


def update_fixation_monitor(eyetracker, prev_sample, last_in_window, consecutive_invalid):
	"""Advance EyeLink fixation monitoring by one frame."""
	if eyetracker is None or not eyetracker.is_active():
		return False, prev_sample, last_in_window, consecutive_invalid, 0.0
	in_window, sample, dist_deg = eyetracker.get_gaze(prev_sample)
	if in_window is None:
		in_window = last_in_window
	else:
		last_in_window = bool(in_window)
	prev_sample = sample
	if dist_deg is None:
		dist_deg = np.inf
	if not in_window:
		consecutive_invalid += 1
		if consecutive_invalid >= max(1, int(EYE['fix_break_consec_frames'])):
			return True, prev_sample, last_in_window, consecutive_invalid, dist_deg
	else:
		consecutive_invalid = 0
	return False, prev_sample, last_in_window, consecutive_invalid, dist_deg


def get_visualizer_draw_size(win):
	"""Return the visible client size for pixel-based visualiser drawing.

	On HiDPI/Retina displays, PsychoPy's win.size can reflect the backing
	framebuffer rather than the visible window area. Drawing with that inflated
	size makes the visualiser appear zoomed-in and clipped. Pyglet's client
	size is the safer coordinate basis for the visualiser layout.
	"""
	try:
		handle = getattr(win, 'winHandle', None)
		if handle is not None and hasattr(handle, 'get_size'):
			w, h = handle.get_size()
			w, h = int(w), int(h)
			if w > 0 and h > 0:
				return w, h
	except Exception:
		pass
	try:
		w, h = win.size
		return int(w), int(h)
	except Exception:
		return 1920, 1080


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

		w, h = get_visualizer_draw_size(self.win)
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
			return 0

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


PART2_SINE_CIRCLE_AMPLITUDE_DEG = 0.7
PART2_SINE_CIRCLE_LOBES = 12
PART2_SINE_CIRCLE_PHASE = 0.0
PART2_ELLIPSE_ASPECT_RATIO = 1.6
PART2_ELLIPSE_ROTATION_RAD = pi / 4.0
PART2_DIAMOND_ROTATION_RAD = 0.0


def build_part2_ellipse_rotations(count):
	rotations = [PART2_ELLIPSE_ROTATION_RAD] * (count // 2)
	rotations.extend([PART2_ELLIPSE_ROTATION_RAD + (np.pi / 2.0)] * (count - len(rotations)))
	random.shuffle(rotations)
	return rotations


def update_part2_theta(theta, speed, direction, dt):
	theta_step = direction * 2 * np.pi * speed * dt
	theta = (theta + theta_step) % (2 * np.pi)
	return theta, theta_step


def part2_circle_radius(radius_deg, theta):
	return radius_deg


def part2_sine_circle_radius(radius_deg, theta, amplitude_deg=PART2_SINE_CIRCLE_AMPLITUDE_DEG,
							 lobes=PART2_SINE_CIRCLE_LOBES, phase=PART2_SINE_CIRCLE_PHASE):
	return radius_deg + amplitude_deg * np.sin(lobes * theta + phase)


def part2_ellipse_axes_matching_circle_circumference(circleRadius, aspectRatio=PART2_ELLIPSE_ASPECT_RATIO):
	aRaw = circleRadius * aspectRatio
	bRaw = circleRadius / aspectRatio
	targetCirc = 2 * pi * circleRadius
	h = ((aRaw - bRaw) ** 2) / ((aRaw + bRaw) ** 2)
	rawCirc = pi * (aRaw + bRaw) * (1 + ((3 * h) / (10 + np.sqrt(4 - 3 * h))))
	if rawCirc <= 0:
		return circleRadius, circleRadius
	scale = targetCirc / rawCirc
	return aRaw * scale, bRaw * scale


def part2_ellipse_radius(circleRadius, theta, aspectRatio=PART2_ELLIPSE_ASPECT_RATIO,
						 rotationRad=PART2_ELLIPSE_ROTATION_RAD):
	a, b = part2_ellipse_axes_matching_circle_circumference(circleRadius, aspectRatio)
	delta = theta - rotationRad
	denominator = np.sqrt((b ** 2) * (np.cos(delta) ** 2) + (a ** 2) * (np.sin(delta) ** 2))
	if denominator <= 0:
		return circleRadius
	return (a * b) / denominator


def part2_diamond_radius(radius_deg, theta, rotationRad=PART2_DIAMOND_ROTATION_RAD):
	delta = theta - rotationRad
	denominator = abs(np.cos(delta)) + abs(np.sin(delta))
	if denominator <= 0:
		return radius_deg
	return radius_deg / denominator


def part2_trajectory_radius(basicShape, radius_deg, theta, ellipse_rotation_rad=None):
	if basicShape == 'circle':
		return part2_circle_radius(radius_deg, theta)
	if basicShape == 'diamond':
		return part2_diamond_radius(radius_deg, theta)
	if basicShape == 'sine_circle':
		return part2_sine_circle_radius(radius_deg, theta)
	if basicShape == 'ellipse':
		rotation_rad = PART2_ELLIPSE_ROTATION_RAD if ellipse_rotation_rad is None else ellipse_rotation_rad
		return part2_ellipse_radius(radius_deg, theta, rotationRad=rotation_rad)
	return part2_circle_radius(radius_deg, theta)


def part2_trajectory_xy(basicShape, radius_deg, theta, ellipse_rotation_rad=None):
	radius = part2_trajectory_radius(basicShape, radius_deg, theta, ellipse_rotation_rad=ellipse_rotation_rad)
	return radius * cos(theta), radius * sin(theta), radius


def part2_motion_frame(basicShape, radius_deg, theta_target, speed, direction, dt, ellipse_rotation_rad=None):
	theta_target, theta_step = update_part2_theta(theta_target, speed, direction, dt)
	theta_distractor = (theta_target + np.pi) % (2 * np.pi)
	x1, y1, radius1 = part2_trajectory_xy(basicShape, radius_deg, theta_target, ellipse_rotation_rad=ellipse_rotation_rad)
	x2, y2, radius2 = part2_trajectory_xy(basicShape, radius_deg, theta_distractor, ellipse_rotation_rad=ellipse_rotation_rad)
	return theta_target, theta_distractor, theta_step, x1, y1, radius1, x2, y2, radius2

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
	r = part2_sine_circle_radius(radius_deg, angle_rad)
	return r * cos(angle_rad), r * sin(angle_rad)

def ellipse_xy(circleRadius, angle_rad):
	r = part2_ellipse_radius(circleRadius, angle_rad)
	return r * cos(angle_rad), r * sin(angle_rad)

def diamond_xy(radius_deg, angle_rad):
	r = part2_diamond_radius(radius_deg, angle_rad)
	return r * cos(angle_rad), r * sin(angle_rad)


def draw_trajectory(myWin, basicShape, radius_deg, cx, cy, num_points=120, ellipse_rotation_rad=None):
	"""
	Draw the actual trajectory (white line) that objects traveled during the trial.
	"""
	trajectory_points = []
	for i in range(num_points):
		angle_rad = 2 * np.pi * i / num_points
		x, y, _ = part2_trajectory_xy(basicShape, radius_deg, angle_rad, ellipse_rotation_rad=ellipse_rotation_rad)
		
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


def validate_part2_motion_rules():
	"""Cheap consistency checks for the Part 2 polar-motion helpers."""
	shape_list = ['circle', 'diamond', 'sine_circle', 'ellipse']
	base_theta = 0.37
	base_radius = 6.0
	sample_dt = 1.0 / 120.0
	speed = 1.0

	_, theta_step = update_part2_theta(base_theta, speed, 1, sample_dt)
	if not np.isclose(theta_step / sample_dt, 2 * np.pi * speed):
		raise AssertionError('Part 2 angular increment does not equal 2πf')

	theta = base_theta
	for _ in range(120):
		theta, _ = update_part2_theta(theta, speed, 1, sample_dt)
	if not np.isclose(((theta - base_theta) % (2 * np.pi)), 0.0, atol=1e-9):
		raise AssertionError('1 rps does not complete one full polar rotation in 1 second')

	for shape in shape_list:
		for theta_sample in np.linspace(0.0, 2 * np.pi, 24, endpoint=False):
			x, y, radius = part2_trajectory_xy(shape, base_radius, theta_sample)
			if not np.isclose(np.hypot(x, y), radius, atol=1e-9):
				raise AssertionError(f'{shape} point is not on the intended radial boundary')

		theta_distractor = (base_theta + np.pi) % (2 * np.pi)
		if not np.isclose(((theta_distractor - base_theta) % (2 * np.pi)), np.pi, atol=1e-12):
			raise AssertionError('Target and distractor are not separated by π radians')

		if shape == 'diamond':
			for corner in (0.0, np.pi / 2.0, np.pi, 3 * np.pi / 2.0):
				eps = 1e-6
				x_lo, y_lo, _ = part2_trajectory_xy(shape, base_radius, corner - eps)
				x_hi, y_hi, _ = part2_trajectory_xy(shape, base_radius, corner + eps)
				if np.hypot(x_hi - x_lo, y_hi - y_lo) > 1e-3:
					raise AssertionError('Diamond trajectory is discontinuous near a corner')

	practice_theta = 1.25
	main_theta = 1.25
	practice_seq = []
	main_seq = []
	for _ in range(10):
		practice_theta, _, _, px, py, pr, _, _, _ = part2_motion_frame('ellipse', base_radius, practice_theta, 0.75, 1, sample_dt)
		main_theta, _, _, mx, my, mr, _, _, _ = part2_motion_frame('ellipse', base_radius, main_theta, 0.75, 1, sample_dt)
		practice_seq.append((practice_theta, px, py, pr))
		main_seq.append((main_theta, mx, my, mr))
	if not np.allclose(np.asarray(practice_seq), np.asarray(main_seq), atol=1e-12):
		raise AssertionError('Practice and main Part 2 motion do not match')


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
					minVal=display_speed_to_stair_value(2.5),
					maxVal=display_speed_to_stair_value(0.2),
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
	part2_motion_ok = True
	part2_motion_error = None
	try:
		validate_part2_motion_rules()
	except Exception as exc:
		part2_motion_ok = False
		part2_motion_error = str(exc)

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
	print('Part 2 polar-motion validation passed:', part2_motion_ok)
	if not part2_motion_ok:
		print('  Part 2 validation error:', part2_motion_error)
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
	if not part2_motion_ok:
		issues.append(part2_motion_error or 'Part 2 motion validation failed')

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

	# -------------------- Pre-experiment screen setup/check --------------------
	screen_settings = run_pre_experiment_screen_check(refreshRate)
	exp_screen_index = screen_settings['exp_screen_index']
	viz_screen_index = screen_settings['viz_screen_index']
	exp_win_size = screen_settings['exp_win_size']
	viz_win_size = screen_settings['viz_win_size']
	fullscr = screen_settings['fullscr']
	refreshRate = screen_settings['refreshRate']
	longFrameLimit = round(1000.0 / refreshRate * (1.0 + longerThanRefreshTolerance), 3)
	print('Using experiment screen:', exp_screen_index)
	print('Using visualizer screen:', viz_screen_index)
	print('Using visualizer window size:', viz_win_size)
	print('Using refreshRate:', refreshRate)
	print('Using longFrameLimit:', longFrameLimit)

	# -------------------- Data file setup --------------------
	subject = 'temp'
	session = 'a'
	dlgLabelsOrdered = []
	myDlg = gui.Dlg(title='MOT comprehensive experiment', pos=(200, 400))
	myDlg.addField('Subject name or ID:', subject, tip='')
	dlgLabelsOrdered.append('subject')
	myDlg.addField('session:', session, tip='a,b,c,')
	dlgLabelsOrdered.append('session')
	myDlg.addText('To abort, press ESC at a trial response screen', color='DimGrey')
	myDlg.show()
	if myDlg.OK:
		thisInfo = myDlg.data
		name = thisInfo[dlgLabelsOrdered.index('subject')]
		if len(name) > 0:
			subject = name
		sessionEntered = thisInfo[dlgLabelsOrdered.index('session')]
		session = str(sessionEntered)
	else:
		print('User cancelled from dialog box.')
		core.quit()

	timeAndDateStr = time.strftime("%d%b%Y_%H-%M", time.localtime())
	EDF_fname_local = None
	if os.path.isdir('.'+os.sep+'dataRaw'):
		dataDir='dataRaw'
	else:
		print('"dataRaw" directory does not exist, so saving data in present working directory')
		dataDir='.'
	datafileName = dataDir+'/'+subject+ '_' + str(session) + '_MOT_pilot_'+timeAndDateStr
	dataFile = open(datafileName+'.tsv', 'w')
	tracker = EyeLinkSession(
		mode=EYE['mode'],
		edf_basename=f'EyeTrack_{subject}_{session}_{timeAndDateStr}',
		host_ip=EYE['host_ip'],
		calibration_type=EYE['calibration_type'],
		log_edf=EYE['log_edf'],
	)
	try:
		tracker.open()
	except RuntimeError as e:
		print('EyeLink open() failed:', e)
		tracker = EyeLinkSession(mode='off', edf_basename='MOT_off', host_ip=EYE['host_ip'], calibration_type=EYE['calibration_type'], log_edf=False)

	# -------------------- Full experiment run (display, timing, response collection) --------------------
	# Minimal window setup (matches practice units)
	try:
		myWin = visual.Window(
			size=exp_win_size,
			screen=exp_screen_index,
			units='deg',
			fullscr=fullscr,
			color=(0, 0, 0),
			monitor=configure_default_monitor(exp_win_size),
			checkTiming=False
		)
		print("Actual experiment window size:", myWin.size)
	except Exception as e:
		print('Warning creating Window:', e)
		print('Creating temporary Monitor specification and retrying Window...')
		try:
			tempMon = monitors.Monitor('tempMonitor', width=38.0, distance=57.0)
			tempMon.setSizePix(exp_win_size)
			myWin = visual.Window(
				size=exp_win_size,
				screen=exp_screen_index,
				units='deg',
				fullscr=fullscr,
				color=(0, 0, 0),
				monitor=tempMon,
				checkTiming=False
			)
			print("Actual fallback experiment window size:", myWin.size)
		except Exception as e2:
			print('Failed to create Window with temporary monitor:', e2)
			raise

	# Stimuli
	fixation = visual.Circle(myWin, radius=0.25, fillColor=(0.9, 0.9, 0.9), lineColor=(0.9, 0.9, 0.9), edges=64, autoLog=False)
	fixationBlank = visual.Circle(myWin, radius=0.25, fillColor=(-1, -1, -1), lineColor=(-1, -1, -1), edges=64, autoLog=False)
	fixationPoint = visual.Circle(myWin, radius=0.08, fillColor=(1, 1, 1), lineColor=None, autoLog=False)
	blobStim = visual.Circle(myWin, radius=0.6, fillColor=(1, -1, -1), lineColor=None)
	blobStim2 = visual.Circle(myWin, radius=0.6, fillColor=(-1, 1, -1), lineColor=None)
	myWin.mouseVisible = True
	stairWin = create_staircase_window(screen_index=viz_screen_index, fullscr=fullscr, win_size=viz_win_size)
	stairViz = StaircaseVisualizer(stairWin) if stairWin is not None else None
	show_staircase_startup(stairWin)
	tracker.setup_graphics(myWin, fix_window_deg=EYE['fixation_window_deg'])
	if tracker.is_active():
		show_text_screen(
			myWin,
			[
				'Eye tracker setup',
				'',
				'Make sure the participant is seated and the tracker camera has a clear view.',
				'',
				'Press SPACE to continue to calibration.',
			],
		)
		show_text_screen(
			myWin,
			[
				'Eye tracker calibration',
				'',
				'Press Space -> Enter to start calibration',
				'',
				'Press Escape here when calibration is complete.',
			],
		)
		tracker.calibrate()

	# Trial timing
	cueFrames = int(refreshRate * trackingExtraTime)
	widthPix = int(myWin.size[0])
	heightPix = int(myWin.size[1])
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
	EDF_fname_local = tracker.edf_filename if tracker is not None and tracker.is_active() else None
	if EDF_fname_local is not None:
		logging.info(f'Eye-tracking enabled; EDF filename={EDF_fname_local}')
	speed_staircases = build_speed_staircases()

	# Write comprehensive header to TSV file (matching MOT Circular format)
	# Column order is explicit for downstream analysis
	header_columns = [
		'trialnum', 'subject', 'session', 'part', 'condition', 'basicShape', 'numObjects', 'speed', 'motionRule', 'trialKind',
		'staircaseIndex', 'staircaseWithinCondition', 'stairKey', 'stairStartSpeed', 'stairValue',
		'initialAngle', 'initialOtherAngle', 'cueFrames', 'timingCheckFrames',
		'eyetrackingEnabled', 'fixatnPeriodFrames', 'trajectorySide', 'trajectoryOffsetDeg', 'ellipseRotationRad', 'correct', 'trialOutcome', 'trialExcluded', 'fixBreakFrame', 'fixBreakDistanceDeg', 'trialDurTotal', 'numTargets', 'whichIsTarget',
		'reversal_count', 'timingBlips', 'numLongFramesAfterCue'
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

	def show_eye_tracker_practice_screen():
		"""Give the participant a final eye-tracker handoff before practice starts."""
		if not tracker.is_active():
			return
		show_text_screen(
			myWin,
			[
				'Eye tracker check',
				'',
				'Keep looking at the central fixation dot.',
				'',
				'Press SPACE to begin the practice trials.',
			],
		)
		tracker.drift_correct()

	def get_balanced_trajectory_side(offset_side_counts, condition):
		"""Assign left/right sides for off-fixation Part 1 trials while keeping the split balanced."""
		if condition not in ('near_displaced', 'far_displaced'):
			return 'center'
		left_count = offset_side_counts.get('left', 0)
		right_count = offset_side_counts.get('right', 0)
		if left_count == right_count:
			side = 'left' if random.random() < 0.5 else 'right'
		elif left_count < right_count:
			side = 'left'
		else:
			side = 'right'
		offset_side_counts[side] += 1
		return side

	def run_part(part_name, trial_list):
		print(f"Running {part_name} with {len(trial_list)} trials")
		offset_side_counts = {'left': 0, 'right': 0}
		
		# Build and run practice trials first
		part_specs = get_part_trial_specs(part_name)
		practice_trials = make_practice_trials(part_name, part_specs)
		if practice_trials:
			show_eye_tracker_practice_screen()
			practice_round = 1
			while True:
				print(f"Running {len(practice_trials)} practice trials for {part_name}")
				show_break_screen(f"{part_name} Practice Trials\n\nYou will now do {len(practice_trials)} practice trials to familiarize yourself with the task.\n\nPress SPACE to begin.")
				# Run practice trials (they don't affect the staircase)
				for pt_idx, practice_trial in enumerate(practice_trials):
					# All practice code here (same as main loop but marked as practice)
					cond = practice_trial['condition']
					motion_rule = practice_trial.get('motionRule', 'standard')
					speed = practice_trial['speed']
					
					if part_name == 'Part1':
						trajectory_side = 'center'
						offset_deg = 0.0
						if cond == 'centred':
							cx = trajectory_center_x_deg[0]
						elif cond == 'near_displaced':
							offset_deg = trajectory_center_x_deg[1]
							trajectory_side = get_balanced_trajectory_side(offset_side_counts, cond)
							cx = offset_deg if trajectory_side == 'right' else -offset_deg
						elif cond == 'far_displaced':
							offset_deg = trajectory_center_x_deg[2]
							trajectory_side = get_balanced_trajectory_side(offset_side_counts, cond)
							cx = offset_deg if trajectory_side == 'right' else -offset_deg
						cy = 0.0
						basicShape = 'circle'
					else:  # Part2 shapes
						cx = 0.0
						cy = 0.0
						basicShape = cond
						trajectory_side = 'center'
						offset_deg = 0.0
					
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
					trialnum = len(results)
					trial_start_time = trialClock.getTime()
					ts = []
					trial_excluded = False
					trial_outcome = 'ok'
					fix_break_frame = -999
					fix_break_distance_deg = -999.0
					prev_sample = None
					last_in_window = True
					consecutive_invalid = 0
					fixatnMinDur = 0.8
					fixatnVariableDur = 0.5
					fixatnPeriodFrames = int((fixatnMinDur + random.random() * fixatnVariableDur) * refreshRate)
					trialDurTotal = trialDurFrames / refreshRate
					tracker.start_recording(trialnum, calib_trial=(trialnum == 0), widthPix=widthPix, heightPix=heightPix)
					tracker.send_message(f'trial_start={trialnum}')
					tracker.send_message(f'trialDurTotal={trialDurTotal}')
					tracker.send_message(f'fixation_period_frames={fixatnPeriodFrames}')
					for fix_frame in range(fixatnPeriodFrames):
						draw_center_fixation(myWin, fixation, fixationBlank, fixationPoint, fix_frame)
						myWin.flip()
					tracker.send_message(f'fixation_complete_trial={trialnum}')
					tracker.send_message(f'Fixation pre-stimulus period of {fixatnPeriodFrames * refreshRate} now ending for trialnum={trialnum}')
					
					# Frame loop (identical to main loop)
					part2_motion_log_enabled = (
						part_name == 'Part2'
						and practice_trial.get('trialKind', 'practice') != 'staircase'
					)
					prev_motion_theta = currTargetAngle
					prev_motion_xy = None
					for frameN in range(trialDurFrames):
						timeSec = frameN / refreshRate
						if part_name == 'Part1' and basicShape == 'circle' and motion_rule == 'varying':
							dt = 1.0 / refreshRate
							# Preserve the signed horizontal offset so left- and right-shifted
							# trajectories use the correct phase-dependent speed profile.
							l = float(cx)
							base_rps = speed
							r = radii[0]
							
							phi_dot_0 = phi_dot_log_eccentricity_base_speed(currPhi[0], base_rps, l, r)
							currPhi[0] = (currPhi[0] + direction * phi_dot_0 * dt) % (2 * np.pi)
							phi_dot_1 = phi_dot_log_eccentricity_base_speed(currPhi[1], base_rps, l, r)
							currPhi[1] = (currPhi[1] + direction * phi_dot_1 * dt) % (2 * np.pi)
							
							x1, y1 = circle_xy(r, currPhi[0], timeSec, base_rps, 0)
							x2, y2 = circle_xy(r, currPhi[1], timeSec, base_rps, 0)
						else:
							dt = 1.0 / refreshRate
							currTargetAngle, distractorAngle, theta_step, x1, y1, radius1, x2, y2, radius2 = part2_motion_frame(
								basicShape,
								radii[0],
								currTargetAngle,
								speed,
								direction,
								dt,
							)
							if part2_motion_log_enabled:
								if prev_motion_xy is None:
									linear_disp_dva_s = 0.0
								else:
									linear_disp_dva_s = np.hypot(x1 - prev_motion_xy[0], y1 - prev_motion_xy[1]) / dt
								theta_delta = ((currTargetAngle - prev_motion_theta + np.pi) % (2 * np.pi)) - np.pi
								print(
									f"PART2_MOTION_LOG practice trial={trialnum} condition={basicShape} nominal_rps={speed:.6f} "
									f"theta={currTargetAngle:.6f} dtheta={theta_delta:.6f} dtheta_dt={(theta_step / dt):.6f} "
									f"radius={radius1:.6f} xy=({x1:.6f},{y1:.6f}) linear_dva_s={linear_disp_dva_s:.6f}"
								)
								prev_motion_theta = currTargetAngle
								prev_motion_xy = (x1, y1)

						fix_broke, prev_sample, last_in_window, consecutive_invalid, dist_deg = update_fixation_monitor(
							tracker, prev_sample, last_in_window, consecutive_invalid
						)
						if fix_broke:
							show_text_screen(
								myWin,
								[
									'Fixation off.',
									'',
									'Return to the central fixation dot.',
									'',
									'Press Space to restart this trial.',
								],
							)
							trial_excluded = True
							trial_outcome = 'fixation_break'
							fix_break_frame = frameN
							fix_break_distance_deg = float(dist_deg if dist_deg is not None else -999.0)
							tracker.stop_trial(trialnum, result_code=1)
							practice_trials.insert(pt_idx, practice_trial)
							break
						
						if next_reversal_idx < len(reversal_times) and timeSec > reversal_times[next_reversal_idx]:
							direction *= -1
							next_reversal_idx += 1
						
						x1 += cx; y1 += cy
						x2 += cx; y2 += cy
						
						draw_center_fixation(myWin, fixation, fixationBlank, fixationPoint, frameN)
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

					if trial_excluded:
						continue

					if trial_excluded:
						show_text_screen(
							myWin,
							[
								'Fixation off.',
								'',
								'Please keep looking at the central fixation dot.',
								'',
								'Press Space to continue.',
							],
						)

					if trial_excluded:
						tracker.stop_trial(trialnum, result_code=1)
						trialDurTotal = trialClock.getTime() - trial_start_time
						reversal_str = '\t'.join([str(round(r, 4)) for r in reversal_times])
						if len(reversal_times) < 10:
							reversal_str += '\t' + '\t'.join(['-999'] * (10 - len(reversal_times)))
						print(trialnum, subject, session, part_name, cond, basicShape, 2, speed, motion_rule, 'practice',
							 0, 'n/a', '-999', -999, -999,
							 round(initialAngle, 4), round(initialOtherAngle, 4),
							 cueFrames, timingCheckFrames,
							 int(tracker.is_active()), fixatnPeriodFrames, trajectory_side, round(abs(offset_deg), 4), round(float(ellipse_rotation_rad), 4) if ellipse_rotation_rad is not None else -999, -999, trial_outcome, 1, fix_break_frame, round(float(fix_break_distance_deg), 4), round(trialDurTotal, 3), 1, target_idx,
							 len(reversal_times), 0, 0,
							 sep='\t', end='\t', file=dataFile)
						print(reversal_str, file=dataFile)
						dataFile.flush()
						results.append({'part': part_name, 'condition': cond, 'motionRule': motion_rule, 'trialKind': 'practice', 'speed': speed, 'stairValue': -999, 'staircaseIndex': 0, 'staircaseWithinCondition': 'n/a', 'stairKey': '-999', 'correct': False, 'excluded': True, 'trialOutcome': trial_outcome, 'trajectorySide': trajectory_side, 'trajectoryOffsetDeg': round(abs(offset_deg), 4)})
						continue
					
					tracker.send_message(f'response_prompt_trial={trialnum}')
					# Response collection
					resp = None
					correct = False
					mouse = event.Mouse(win=myWin)
					try:
						mouse.setVisible(True)
					except Exception:
						myWin.mouseVisible = True
					clicked = False
					if autoAdvance:
						clicked = True
						correct = True
						draw_center_fixation(myWin, fixation, fixationBlank, fixationPoint)
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
							draw_center_fixation(myWin, fixation, fixationBlank, fixationPoint)
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
					
					tracker.send_message(f'trial_end={trialnum}')
					tracker.stop_recording()
					# Write practice trial to TSV
					trialDurTotal = trialClock.getTime() - trial_start_time
					reversal_str = '\t'.join([str(round(r, 4)) for r in reversal_times])
					if len(reversal_times) < 10:
						reversal_str += '\t' + '\t'.join(['-999'] * (10 - len(reversal_times)))
					
					print(trialnum, subject, session, part_name, cond, basicShape, 2, speed, motion_rule, 'practice',
						  0, 'n/a', '-999', -999, -999,
						  round(initialAngle, 4), round(initialOtherAngle, 4),
						  cueFrames, timingCheckFrames,
							int(tracker.is_active()), fixatnPeriodFrames, trajectory_side, round(abs(offset_deg), 4), round(float(ellipse_rotation_rad), 4) if ellipse_rotation_rad is not None else -999, int(correct), round(trialDurTotal, 3), 1, target_idx,
					      len(reversal_times), timing_blips, num_long_frames_after_cue,
						  sep='\t', end='\t', file=dataFile)
					print(reversal_str, file=dataFile)
					dataFile.flush()
					results.append({'part': part_name, 'condition': cond, 'motionRule': motion_rule, 'trialKind': 'practice', 'speed': speed, 'stairValue': -999, 'staircaseIndex': 0, 'staircaseWithinCondition': 'n/a', 'stairKey': '-999', 'correct': bool(correct), 'trajectorySide': trajectory_side, 'trajectoryOffsetDeg': round(abs(offset_deg), 4)})
			
				restart_practice = False
				practice_done_text = visual.TextStim(
					myWin,
					text=f"Practice trials complete for {part_name}.\n\nPress R to repeat the practice trials.\nPress SPACE to begin the actual {part_name} trials.",
					color=(1, 1, 1),
					height=0.8,
					wrapWidth=40,
				)
				practice_done_text.draw()
				myWin.flip()
				while True:
					keys = event.waitKeys(keyList=['r', 'space', 'spacebar', 'escape'])
					if not keys:
						continue
					if 'escape' in keys:
						core.quit()
					if 'r' in keys:
						practice_round += 1
						restart_practice = True
						break
					if 'space' in keys or 'spacebar' in keys:
						break
				if not restart_practice:
					break
		
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
				trajectory_side = 'center'
				offset_deg = 0.0
				if cond == 'centred':
					cx = trajectory_center_x_deg[0]
				elif cond == 'near_displaced':
					offset_deg = trajectory_center_x_deg[1]
					trajectory_side = get_balanced_trajectory_side(offset_side_counts, cond)
					cx = offset_deg if trajectory_side == 'right' else -offset_deg
				elif cond == 'far_displaced':
					offset_deg = trajectory_center_x_deg[2]
					trajectory_side = get_balanced_trajectory_side(offset_side_counts, cond)
					cx = offset_deg if trajectory_side == 'right' else -offset_deg
				cy = 0.0
				basicShape = 'circle'
			else:  # Part2 shapes
				cx = 0.0
				cy = 0.0
				basicShape = cond  # 'diamond','sine_circle','ellipse'
				trajectory_side = 'center'
				offset_deg = 0.0

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
			trialnum = len(results)
			trial_start_time = trialClock.getTime()
			ts = []
			fixatnMinDur = 0.8
			fixatnVariableDur = 0.5
			fixatnPeriodFrames = int((fixatnMinDur + random.random() * fixatnVariableDur) * refreshRate)
			trialDurTotal = trialDurFrames / refreshRate
			tracker.start_recording(trialnum, calib_trial=(trialnum == 0), widthPix=widthPix, heightPix=heightPix)
			tracker.send_message(f'trial_start={trialnum}')
			tracker.send_message(f'trialDurTotal={trialDurTotal}')
			tracker.send_message(f'fixation_period_frames={fixatnPeriodFrames}')
			for fix_frame in range(fixatnPeriodFrames):
				draw_center_fixation(myWin, fixation, fixationBlank, fixationPoint, fix_frame)
				myWin.flip()
			tracker.send_message(f'fixation_complete_trial={trialnum}')
			tracker.send_message(f'Fixation pre-stimulus period of {fixatnPeriodFrames * refreshRate} now ending for trialnum={trialnum}')

			# frame loop
			trial_excluded = False
			trial_outcome = 'ok'
			fix_break_frame = -999
			fix_break_distance_deg = -999.0
			prev_sample = None
			last_in_window = True
			consecutive_invalid = 0
			part2_motion_log_enabled = (
				part_name == 'Part2'
				and trial_kind != 'staircase'
			)
			prev_motion_theta = currTargetAngle
			prev_motion_xy = None
			for frameN in range(trialDurFrames):
				timeSec = frameN / refreshRate
				if part_name == 'Part1' and basicShape == 'circle' and motion_rule == 'varying':
					# Part 1: preserve the signed horizontal offset so left- and right-shifted
					# trajectories use the correct phase-dependent speed profile.
					dt = 1.0 / refreshRate
					l = float(cx)
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
					dt = 1.0 / refreshRate
					currTargetAngle, distractorAngle, theta_step, x1, y1, radius1, x2, y2, radius2 = part2_motion_frame(
						basicShape,
						radii[0],
						currTargetAngle,
						speed,
						direction,
						dt,
					)
					if part2_motion_log_enabled:
						if prev_motion_xy is None:
							linear_disp_dva_s = 0.0
						else:
							linear_disp_dva_s = np.hypot(x1 - prev_motion_xy[0], y1 - prev_motion_xy[1]) / dt
						theta_delta = ((currTargetAngle - prev_motion_theta + np.pi) % (2 * np.pi)) - np.pi
						print(
							f"PART2_MOTION_LOG main trial={trialnum} kind={trial_kind} condition={basicShape} nominal_rps={speed:.6f} "
							f"theta={currTargetAngle:.6f} dtheta={theta_delta:.6f} dtheta_dt={(theta_step / dt):.6f} "
							f"radius={radius1:.6f} xy=({x1:.6f},{y1:.6f}) linear_dva_s={linear_disp_dva_s:.6f}"
						)
						prev_motion_theta = currTargetAngle
						prev_motion_xy = (x1, y1)

						fix_broke, prev_sample, last_in_window, consecutive_invalid, dist_deg = update_fixation_monitor(
							tracker, prev_sample, last_in_window, consecutive_invalid
						)
						if fix_broke:
							show_text_screen(
								myWin,
								[
									'Fixation off.',
									'',
									'Return to the central fixation dot.',
									'',
									'Press Space to restart this trial.',
								],
							)
							trial_excluded = True
							trial_outcome = 'fixation_break'
							fix_break_frame = frameN
							fix_break_distance_deg = float(dist_deg if dist_deg is not None else -999.0)
							tracker.stop_trial(trialnum, result_code=1)
							trial_list.insert(ti, thisTrial)
							break

				if next_reversal_idx < len(reversal_times) and timeSec > reversal_times[next_reversal_idx]:
					direction *= -1
					next_reversal_idx += 1

				# offset from fixation
				x1 += cx; y1 += cy
				x2 += cx; y2 += cy

				# draw everything for this frame
				draw_center_fixation(myWin, fixation, fixationBlank, fixationPoint, frameN)
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

			if trial_excluded:
				continue

			if trial_excluded:
				tracker.stop_trial(trialnum, result_code=1)
				trialDurTotal = trialClock.getTime() - trial_start_time
				reversal_str = '\t'.join([str(round(r, 4)) for r in reversal_times])
				if len(reversal_times) < 10:
					reversal_str += '\t' + '\t'.join(['-999'] * (10 - len(reversal_times)))
				print(trialnum, subject, session, part_name, cond, basicShape, 2, speed, motion_rule, trial_kind,
					staircase_index, staircase_label, stair_key, stair_start_speed, stair_value,
					round(initialAngle, 4), round(initialOtherAngle, 4),
					cueFrames, timingCheckFrames,
					int(tracker.is_active()), fixatnPeriodFrames, trajectory_side, round(abs(offset_deg), 4), round(float(ellipse_rotation_rad), 4) if ellipse_rotation_rad is not None else -999, -999, trial_outcome, 1, fix_break_frame, round(float(fix_break_distance_deg), 4), round(trialDurTotal, 3), 1, target_idx,
					len(reversal_times), 0, 0,
					sep='\t', end='\t', file=dataFile)
				print(reversal_str, file=dataFile)
				dataFile.flush()
				results.append({'part': part_name, 'condition': cond, 'motionRule': motion_rule, 'trialKind': trial_kind, 'speed': speed, 'stairValue': stair_value, 'staircaseIndex': staircase_index, 'staircaseWithinCondition': staircase_label, 'stairKey': stair_key, 'correct': False, 'excluded': True, 'trialOutcome': trial_outcome, 'trajectorySide': trajectory_side, 'trajectoryOffsetDeg': round(abs(offset_deg), 4)})
				continue

			tracker.send_message(f'response_prompt_trial={trialnum}')
			# response collection: ask participant to click the target
			# draw the objects continuously while waiting so they remain visible
			resp = None
			correct = False
			mouse = event.Mouse(win=myWin)
			try:
				mouse.setVisible(True)
			except Exception:
				myWin.mouseVisible = True
			clicked = False
			if autoAdvance:
				clicked = True
				correct = True
				draw_center_fixation(myWin, fixation, fixationBlank, fixationPoint)
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
					draw_center_fixation(myWin, fixation, fixationBlank, fixationPoint)
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
				  cueFrames, timingCheckFrames,
					int(tracker.is_active()), fixatnPeriodFrames, trajectory_side, round(abs(offset_deg), 4), round(float(ellipse_rotation_rad), 4) if ellipse_rotation_rad is not None else -999, int(correct), round(trialDurTotal, 3), 1, target_idx,
				  len(reversal_times), timing_blips, num_long_frames_after_cue,
				  sep='\t', end='\t', file=dataFile)
			print(reversal_str, file=dataFile)
			dataFile.flush()
			results.append({'part': part_name, 'condition': cond, 'motionRule': motion_rule, 'trialKind': trial_kind, 'speed': speed, 'stairValue': stair_value, 'staircaseIndex': staircase_index, 'staircaseWithinCondition': staircase_label, 'stairKey': stair_key, 'correct': bool(correct), 'trajectorySide': trajectory_side, 'trajectoryOffsetDeg': round(abs(offset_deg), 4)})

	if 'part1' in parts_to_run:
		run_part('Part1', trials['part1'])
		# provide a break after Part 1 completes
		show_break_screen('Part 1 complete. Please take a longer break if needed. Press SPACE to continue to Part 2.')
	if 'part2' in parts_to_run:
		run_part('Part2', trials['part2'])

	show_text_screen(
		myWin,
		[
			'Thank you for taking part in the study.',
			'',
			'This is the end of the study.',
			'',
			'Please notify your experimenter now.',
		],
	)

	if tracker is not None:
		try:
			tracker.close(output_dir=dataDir, base_name=f'{subject}_{session}_{timeAndDateStr}')
		except Exception as e:
			logging.info(f'EyeLink close error: {e}')

	# Final summary
	n_correct = sum(1 for r in results if r['correct'] and not r.get('excluded', False))
	n_excluded = sum(1 for r in results if r.get('excluded', False))
	print('\nExperiment complete. Total correct:', n_correct, 'out of', len(results))
	print('Excluded trials:', n_excluded)
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
