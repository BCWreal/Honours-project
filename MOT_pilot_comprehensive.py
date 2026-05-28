"""
MOT_pilot_comprehensive.py

Three-part experiment scaffold derived from MOT Circular / practice files.
COMPREHENSIVE VERSION with detailed results logging matching MOT Circular.py

Parts:
 - Part 1: Off-Fixation Standard / Varying Speed block
 - Part 2: Different Shapes (diamond, sine-circle, ellipse)

General rules enforced:
 - 2 objects per trial (1 target, 1 distractor)
 - Adaptive staircase on speed across a 1.5 to 2.2 rps range, 96 trials/condition
 - 5 fixed 1 rps attention-check trials per condition
 - Part 1: 5 conditions total = 505 trials
 - Part 2: 3 conditions = 303 trials
 - 808 total trials
 - Circular trajectories use radius 6 deg
 - Feedback sounds: correct / incorrect

Results logging includes: trial details, initial angles, reversal times, trial duration
"""

from psychopy import prefs
prefs.hardware['audioLib'] = ['pygame']
from psychopy import visual, core, event, sound, gui, monitors, data
import numpy as np, random, time, os
from math import pi, cos, sin
import itertools
import sys
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


# -------------------- Experiment parameters (match practice files) --------------------
numRings = 1
radii = np.array([6.0])  # deg, ensure circular trajectories use 6 degrees
respRadius = radii[0]
units = 'deg'

refreshRate = 100.0
trialDurMin = 2
trackingExtraTime = 1.2
trackVariableIntervMax = 2.5
autoAdvance = False

stair_nUp = 1
stair_nDown = 3
stair_stepSizes = [.3, .3, .2, .1, .1, .05]
stair_start = 1.85
stair_min = 1.5
stair_max = 2.2
stair_trials_per_condition = 96
attention_check_trials_per_condition = 5
attention_check_speed = 1.0


def stair_value_to_speed(stair_value):
	"""Map the StairHandler value so speed increases when StairHandler steps down.

	This makes the trial speed increase after 3 consecutive correct responses and
	decrease immediately after an incorrect response.
	"""
	return stair_min + stair_max - stair_value


def make_attention_check_trials(part_name, trial_specs):
	trials = []
	for spec in trial_specs:
		for rep in range(attention_check_trials_per_condition):
			trials.append({
				'part': part_name,
				'condition': spec['condition'],
				'motionRule': spec.get('motionRule', 'standard'),
				'trialKind': 'attention_check',
				'speed': attention_check_speed,
			})
	random.shuffle(trials)
	return trials

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


# -------------------- Trial generation / counterbalancing --------------------
def make_stair_trials(part_name, trial_specs):
	trials = []
	for spec in trial_specs:
		for rep in range(stair_trials_per_condition):
			trial = {
				'part': part_name,
				'condition': spec['condition'],
				'motionRule': spec.get('motionRule', 'standard'),
				'stairKey': spec['stairKey'],
			}
			trials.append(trial)
	random.shuffle(trials)
	return trials


def build_session():
	# Every condition uses a staircase-driven base speed.
	combined_trial_specs = [
		{'condition': 'centred', 'motionRule': 'standard', 'stairKey': 'Part1|centred|standard'},
		{'condition': 'near_displaced', 'motionRule': 'standard', 'stairKey': 'Part1|near_displaced|standard'},
		{'condition': 'far_displaced', 'motionRule': 'standard', 'stairKey': 'Part1|far_displaced|standard'},
		{'condition': 'near_displaced', 'motionRule': 'varying', 'stairKey': 'Part1|near_displaced|varying'},
		{'condition': 'far_displaced', 'motionRule': 'varying', 'stairKey': 'Part1|far_displaced|varying'},
	]
	part12_trials = make_stair_trials('Part1', combined_trial_specs)
	part12_trials.extend(make_attention_check_trials('Part1', combined_trial_specs))
	shape_trial_specs = [
		{'condition': 'diamond', 'motionRule': 'shape', 'stairKey': 'Part2|diamond'},
		{'condition': 'sine_circle', 'motionRule': 'shape', 'stairKey': 'Part2|sine_circle'},
		{'condition': 'ellipse', 'motionRule': 'shape', 'stairKey': 'Part2|ellipse'},
	]
	part3_trials = make_stair_trials('Part2', shape_trial_specs)
	part3_trials.extend(make_attention_check_trials('Part2', shape_trial_specs))
	random.shuffle(part12_trials)
	random.shuffle(part3_trials)

	# keep trials separate per part (they are run independently per instructions)
	return {'part1': part12_trials, 'part2': part3_trials}


def build_varying_staircases():
	staircases = {}
	for part_name, trial_specs in [
		('Part1', [
			{'condition': 'centred', 'motionRule': 'standard', 'stairKey': 'Part1|centred|standard'},
			{'condition': 'near_displaced', 'motionRule': 'standard', 'stairKey': 'Part1|near_displaced|standard'},
			{'condition': 'far_displaced', 'motionRule': 'standard', 'stairKey': 'Part1|far_displaced|standard'},
			{'condition': 'near_displaced', 'motionRule': 'varying', 'stairKey': 'Part1|near_displaced|varying'},
			{'condition': 'far_displaced', 'motionRule': 'varying', 'stairKey': 'Part1|far_displaced|varying'},
		]),
		('Part2', [
			{'condition': 'diamond', 'motionRule': 'shape', 'stairKey': 'Part2|diamond'},
			{'condition': 'sine_circle', 'motionRule': 'shape', 'stairKey': 'Part2|sine_circle'},
			{'condition': 'ellipse', 'motionRule': 'shape', 'stairKey': 'Part2|ellipse'},
		]),
	]:
		for spec in trial_specs:
			staircases[spec['stairKey']] = data.StairHandler(
				startVal=stair_start,
			stepType='lin',
				stepSizes=stair_stepSizes,
				minVal=stair_min,
				maxVal=stair_max,
				nUp=stair_nUp,
				nDown=stair_nDown,
				nTrials=stair_trials_per_condition,
				extraInfo={'part': part_name, 'condition': spec['condition'], 'motionRule': spec['motionRule']}
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
	expected_counts = {
		('Part1', 'centred', 'standard'): stair_trials_per_condition + attention_check_trials_per_condition,
		('Part1', 'near_displaced', 'standard'): stair_trials_per_condition + attention_check_trials_per_condition,
		('Part1', 'far_displaced', 'standard'): stair_trials_per_condition + attention_check_trials_per_condition,
		('Part1', 'near_displaced', 'varying'): stair_trials_per_condition + attention_check_trials_per_condition,
		('Part1', 'far_displaced', 'varying'): stair_trials_per_condition + attention_check_trials_per_condition,
		('Part2', 'diamond', 'shape'): stair_trials_per_condition + attention_check_trials_per_condition,
		('Part2', 'sine_circle', 'shape'): stair_trials_per_condition + attention_check_trials_per_condition,
		('Part2', 'ellipse', 'shape'): stair_trials_per_condition + attention_check_trials_per_condition,
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
		# compute expected based on the condition+motionRule split in each part
		unique_keys = set([(t.get('part'), t.get('condition'), t.get('motionRule', 'standard')) for t in trials_by_part[pname]])
		expected = sum(expected_counts.get(key, 0) for key in unique_keys)
		print(f"{pname}: {counts[pname]} trials (expected {expected})")
		print('  speed distribution:', speed_counts[pname])

	total_trials = sum(counts.values())
	# compute expected total using the staircase split above
	expected_total = sum(expected_counts.values())
	print('Total trials across 3 parts =', total_trials, f'(expected {expected_total})')

	print('\nFeedback sounds available: correct/inaccurate assigned')
	print('beep_correct:', beep_correct)
	print('beep_incorrect:', beep_incorrect)

	# Quick assertions to flag any issue
	issues = []
	for pname in counts:
		unique_keys = set([(t.get('part'), t.get('condition'), t.get('motionRule', 'standard')) for t in trials_by_part[pname]])
		expected = sum(expected_counts.get(key, 0) for key in unique_keys)
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
	cueOutline = visual.Circle(myWin, radius=radii[0] + 0.6, lineColor=(1, 1, 1), fillColor=None)

	# Trial timing
	cueFrames = int(refreshRate * trackingExtraTime)
	trialDurFrames = int(trialDurMin * refreshRate) + int(trackingExtraTime * refreshRate)
	trial_duration_sec = trialDurFrames / refreshRate

	# Eccentricities used in practice files
	practice_trajectoryCenterXDeg = [0.0, 5.0, 10.0]
	practice_trajectoryCenterYDeg = [0.0, 0.0, 0.0]

	results = []
	identicalBlobColor = np.array([1, -1, -1])
	targetCueColor = np.array([1, 1, 1])
	# distractor during cue interval (red), flash colors for feedback
	distractorCueColor = np.array([1, -1, -1])
	greenFlashColor = np.array([-1, 1, -1])
	orangeFlashColor = np.array([1, 0.5, -1])
	trialClock = core.Clock()
	speed_staircases = build_speed_staircases()

	# Write comprehensive header to TSV file (matching MOT Circular format)
	# Column order is explicit for downstream analysis
	header_columns = [
		'trialnum', 'subject', 'session', 'part', 'condition', 'basicShape', 'numObjects', 'speed', 'motionRule', 'trialKind',
		'initialAngle', 'initialOtherAngle', 'cueFrames', 'correct', 'trialDurTotal', 'numTargets', 'whichIsTarget',
		'reversal_count'
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
			motion_rule = thisTrial.get('motionRule', 'standard')
			trial_kind = thisTrial.get('trialKind', 'staircase')
			if part_name == 'Part1':
				if cond == 'centred':
					cx = practice_trajectoryCenterXDeg[0]
				elif cond == 'near_displaced':
					cx = practice_trajectoryCenterXDeg[1]
				elif cond == 'far_displaced':
					cx = practice_trajectoryCenterXDeg[2]
				cy = 0.0
				basicShape = 'circle'
			else:  # Part2 shapes
				cx = 0.0
				cy = 0.0
				basicShape = cond  # 'diamond','sine_circle','ellipse'

			if trial_kind == 'attention_check':
				speed = attention_check_speed
				stair_value = None
			else:
				stair_value = speed_staircases[thisTrial['stairKey']].next()
				speed = stair_value_to_speed(stair_value)

			# starting angles (two objects opposite)
			currAngle = random.random() * 2 * pi
			otherAngle = (currAngle + pi) % (2 * pi)
			initialAngle = currAngle
			initialOtherAngle = otherAngle
			
			currPhi = [random.random() * 2 * pi, 0.0]
			currPhi[1] = (currPhi[0] + pi) % (2 * pi)
			direction = 1
			reversal_times = get_reversal_times(trial_duration_sec)
			next_reversal_idx = 0

			# cue which is target: target is object 0
			target_idx = 0

			# Record trial start time
			trial_start_time = trialClock.getTime()

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

					currAngle = (currAngle + angleStep) % (2 * pi)
					otherAngle = (otherAngle + angleStep) % (2 * pi)

					# compute positions depending on shape
					if basicShape == 'circle':
						x1, y1 = circle_xy(radii[0], currAngle, timeSec, speed, 0)
						x2, y2 = circle_xy(radii[0], otherAngle, timeSec, speed, 0)
					elif basicShape == 'diamond':
						x1, y1 = diamond_xy(radii[0], currAngle)
						x2, y2 = diamond_xy(radii[0], otherAngle)
					elif basicShape == 'sine_circle':
						x1, y1 = sine_circle_xy(radii[0], currAngle)
						x2, y2 = sine_circle_xy(radii[0], otherAngle)
					elif basicShape == 'ellipse':
						x1, y1 = ellipse_xy(radii[0], currAngle)
						x2, y2 = ellipse_xy(radii[0], otherAngle)
					else:
						x1, y1 = circle_xy(radii[0], currAngle)
						x2, y2 = circle_xy(radii[0], otherAngle)

				if next_reversal_idx < len(reversal_times) and timeSec > reversal_times[next_reversal_idx]:
					direction *= -1
					next_reversal_idx += 1

				# offset by center
				x1 += cx; y1 += cy
				x2 += cx; y2 += cy

				# draw
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

			if trial_kind != 'attention_check':
				# Standard PsychoPy StairHandler semantics: True=correct, False=incorrect.
				speed_staircases[thisTrial['stairKey']].addResponse(bool(correct))

			# Write comprehensive trial result to data file
			trialnum = len(results)
			trialDurTotal = trialClock.getTime() - trial_start_time
			
			# Format reversal times for output
			reversal_str = '\t'.join([str(round(r, 4)) for r in reversal_times])
			if len(reversal_times) < 10:
				reversal_str += '\t' + '\t'.join(['-999'] * (10 - len(reversal_times)))
			
			# Write all trial data
			print(trialnum, subject, session, part_name, cond, basicShape, 2, speed, motion_rule, trial_kind,
				  round(initialAngle, 4), round(initialOtherAngle, 4),
				  cueFrames, int(correct), round(trialDurTotal, 3), 1, target_idx,
				  len(reversal_times),
				  sep='\t', end='\t', file=dataFile)
			print(reversal_str, file=dataFile)
			dataFile.flush()
			results.append({'part': part_name, 'condition': cond, 'motionRule': motion_rule, 'trialKind': trial_kind, 'speed': speed, 'stairValue': stair_value, 'correct': bool(correct)})

	def run_part1(trial_list):
		# Enforce 2 objects per trial and run Part 1 (Off-Fixation Standard)
		print('\n--- Starting Part 1: Off-Fixation Standard ---')
		# Part 1 uses circular trajectories and fixation at center while trajectory center is offset
		run_part('Part1', trial_list)

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
