"""
MOT_pilot_comprehensive.py

Three-part experiment scaffold derived from MOT Circular / practice files.
COMPREHENSIVE VERSION with detailed results logging matching MOT Circular.py

Parts:
 - Part 1: Off-Fixation Standard (circular trajectories)
 - Part 2: Varying Speed (uses same modulation approach as MOT_Varying_speed)
 - Part 3: Different Shapes (diamond, sine-circle, ellipse)

General rules enforced:
 - 2 objects per trial (1 target, 1 distractor)
 - Speeds: [1.0,1.5,1.7,1.9,2.0,2.2] rps, 16 repeats each = 96 trials/condition
 - 3 conditions/part = 288 trials/part, 864 total
 - Circular trajectories use radius 6 deg
 - Feedback sounds: correct / incorrect

Results logging includes: trial details, initial angles, reversal times, trial duration
"""

from psychopy import prefs
prefs.hardware['audioLib'] = ['pygame']
from psychopy import visual, core, event, sound, gui, monitors
import numpy as np, random, time, os
from math import pi, cos, sin
import itertools
import sys


# -------------------- Experiment parameters (match practice files) --------------------
numRings = 1
radii = np.array([6.0])  # deg, ensure circular trajectories use 6 degrees
respRadius = radii[0]
units = 'deg'

refreshRate = 100.0
trialDurMin = 2
trackingExtraTime = 1.2
trackVariableIntervMax = 2.5
autoAdvance = os.environ.get('MOT_AUTO_ADVANCE', '0') == '1'

speeds_base = [1.0, 1.5, 1.7, 1.9, 2.0, 2.2]  # rps
repeats_per_speed = 16

# Sounds
beep_correct = sound.Sound(value='C', secs=0.08)  # simple tone
beep_incorrect = sound.Sound(value='A', secs=0.12)

timeTillReversalMin = 0.5
timeTillReversalMax = 2.0
badTimingCushion = 0.3
part2_log_polar_V_base = 1.0  # Base speed in log-polar space; scaled by trial speed to match nominal rps
part2_log_polar_ref_speed = 1.0  # Reference speed (rps) for which V_base was calibrated


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
def make_trials_for_part(part_name, conditions):
	# conditions: list of condition identifiers; for parts 1 & 2 these are fixation offsets (e.g., 'centred','near','far')
	trials = []
	for cond in conditions:
		for sp in speeds_base:
			for rep in range(repeats_per_speed):
				trials.append({'part': part_name, 'condition': cond, 'speed': sp})
	random.shuffle(trials)
	return trials


def build_session():
	# Define the three parts and their three conditions each
	fixation_conditions = ['centred', 'near_displaced', 'far_displaced']
	part1_trials = make_trials_for_part('Part1_OffFix_Standard', fixation_conditions)
	part2_trials = make_trials_for_part('Part2_VaryingSpeed', fixation_conditions)
	shape_conditions = ['diamond', 'sine_circle', 'ellipse']
	part3_trials = make_trials_for_part('Part3_DiffShapes', shape_conditions)

	# keep trials separate per part (they are run independently per instructions)
	return {'part1': part1_trials, 'part2': part2_trials, 'part3': part3_trials}


# -------------------- Minimal run loop for verification (non-optimized, non-graphical checks) --------------------
def run_checks_and_report(trials_by_part):
	# Check radii for circular trajectories
	circle_radius_ok = (abs(radii[0] - 6.0) < 1e-6)

	# Trial counts
	counts = {k: len(v) for k, v in trials_by_part.items()}

	# Counterbalancing: count speeds per part
	speed_counts = {}
	for pname, tlist in trials_by_part.items():
		sc = {}
		for t in tlist:
			sc[t['speed']] = sc.get(t['speed'], 0) + 1
		speed_counts[pname] = sc

	print('\n=== MOT_pilot verification report ===')
	print('Circular radii set to 6 deg:', circle_radius_ok)
	for pname in counts:
		# compute expected based on unique conditions and speeds
		unique_conds = set([t['condition'] for t in trials_by_part[pname]])
		expected = len(unique_conds) * len(speeds_base) * repeats_per_speed
		print(f"{pname}: {counts[pname]} trials (expected {expected})")
		print('  speed distribution:', speed_counts[pname])

	total_trials = sum(counts.values())
	expected_total = sum([len(set([t['condition'] for t in trials_by_part[p]])) * len(speeds_base) * repeats_per_speed for p in trials_by_part])
	print('Total trials across 3 parts =', total_trials, f'(expected {expected_total})')

	print('\nFeedback sounds available: correct/inaccurate assigned')
	print('beep_correct:', beep_correct)
	print('beep_incorrect:', beep_incorrect)

	# Quick assertions to flag any issue
	issues = []
	for pname in counts:
		unique_conds = set([t['condition'] for t in trials_by_part[pname]])
		expected = len(unique_conds) * len(speeds_base) * repeats_per_speed
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
	parts_to_run = [part.strip().lower() for part in os.environ.get('MOT_PARTS', 'part1,part2,part3').split(',') if part.strip()]

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
	trialClock = core.Clock()

	# Write comprehensive header to TSV file (matching MOT Circular format)
	header = 'trialnum\tsubject\tsession\tpart\tcondition\tbasicShape\tnumObjects\tspeed'
	header += '\tinitialAngle\tinitialOtherAngle\tcueFrames\tcorrect\ttrialDurTotal\tnumTargets\twhichIsTarget'
	header += '\treversal_count'
	for i in range(10):  # space for up to 10 reversals
		header += f'\treversal_{i}'
	print(header, file=dataFile)

	def run_part(part_name, trial_list):
		print(f"Running {part_name} with {len(trial_list)} trials")
		for ti, thisTrial in enumerate(trial_list):
			# determine center
			cond = thisTrial['condition']
			if part_name in ('part1', 'part2'):
				if cond == 'centred':
					cx = practice_trajectoryCenterXDeg[0]
				elif cond == 'near_displaced':
					cx = practice_trajectoryCenterXDeg[1]
				elif cond == 'far_displaced':
					cx = practice_trajectoryCenterXDeg[2]
				cy = 0.0
				basicShape = 'circle'
			else:  # part3 shapes
				cx = 0.0
				cy = 0.0
				basicShape = cond  # 'diamond','sine_circle','ellipse'

			speed = thisTrial['speed']

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
				if part_name == 'part2' and basicShape == 'circle' and (cx != 0.0 or cy != 0.0):
					# Part 2 displaced conditions: Each object moves independently under log-polar speed equation
					# phi_dot = (V / r) * sqrt(l^2 + r^2 + 2*l*r*cos(phi))
					# V is scaled to maintain correct average revolution time despite position-dependent instantaneous speed
					dt = 1.0 / refreshRate
					l = np.sqrt(cx * cx + cy * cy)
					
					# Compute mean_inv_rho on first frame of this trial
					if frameN == 0:
						mean_inv_rho = compute_mean_inv_rho(l, radii[0])
						V_scaled = speed * radii[0] * 2 * np.pi * mean_inv_rho
						print(f"DEBUG: Part 2 log-polar motion ACTIVE for trial {ti+1}, speed={speed}, l={l:.2f}, mean_inv_rho={mean_inv_rho:.6f}, V_scaled={V_scaled:.3f}")
					
					# Update object 1 (target) independently
					currPhi[0] = update_phi_log_polar(
						currPhi[0],
						V_scaled,
						l,
						radii[0],
						dt,
						direction=direction,
					)
					
					# Update object 2 (distractor) independently
					currPhi[1] = update_phi_log_polar(
						currPhi[1],
						V_scaled,
						l,
						radii[0],
						dt,
						direction=direction,
					)
					
					x1, y1 = circle_xy(radii[0], currPhi[0], timeSec, speed, 0)
					x2, y2 = circle_xy(radii[0], currPhi[1], timeSec, speed, 0)
				else:
					# Parts 1 & 3: Both objects move together at the same angular speed
					if frameN == 0 and part_name == 'part2':
						print(f"DEBUG: Part 2 FALLING THROUGH to constant speed. part_name={part_name}, basicShape={basicShape}, cx={cx}, cy={cy}")
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
				blobStim.setFillColor(identicalBlobColor, log=False)
				blobStim2.setFillColor(identicalBlobColor, log=False)
				blobStim.setLineColor(None, log=False)
				blobStim2.setLineColor(None, log=False)
				blobStim.setPos((x1, y1)); blobStim.draw()
				blobStim2.setPos((x2, y2)); blobStim2.draw()

				# draw cue at start of trial to indicate target
				if frameN < cueFrames:
					if target_idx == 0:
						blobStim.setLineColor(targetCueColor, log=False)
						blobStim.setLineWidth(4)
						blobStim.setPos((x1, y1)); blobStim.draw()
						blobStim.setLineColor(None, log=False)

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
					# Flash selected object green twice
					if target_idx == 0:
						apply_visual_feedback(target_idx, True, x1, y1, blobStim, fixation)
					else:
						apply_visual_feedback(target_idx, True, x2, y2, blobStim2, fixation)
				else:
					beep_incorrect.play()
					# Flash selected object red twice
					if picked == 0:
						apply_visual_feedback(picked, False, x1, y1, blobStim, fixation)
					else:
						apply_visual_feedback(picked, False, x2, y2, blobStim2, fixation)

			# Write comprehensive trial result to data file
			trialnum = len(results)
			trialDurTotal = trialClock.getTime() - trial_start_time
			
			# Format reversal times for output
			reversal_str = '\t'.join([str(round(r, 4)) for r in reversal_times])
			if len(reversal_times) < 10:
				reversal_str += '\t' + '\t'.join(['-999'] * (10 - len(reversal_times)))
			
			# Write all trial data
			print(trialnum, subject, session, part_name, cond, basicShape, 2, speed,
				  round(initialAngle, 4), round(initialOtherAngle, 4),
				  cueFrames, int(correct), round(trialDurTotal, 3), 1, target_idx,
				  len(reversal_times),
				  sep='\t', end='\t', file=dataFile)
			print(reversal_str, file=dataFile)
			dataFile.flush()
			results.append({'part': part_name, 'condition': cond, 'speed': speed, 'correct': bool(correct)})

	def run_part1(trial_list):
		# Enforce 2 objects per trial and run Part 1 (Off-Fixation Standard)
		print('\n--- Starting Part 1: Off-Fixation Standard ---')
		# Part 1 uses circular trajectories and fixation at center while trajectory center is offset
		run_part('part1', trial_list)

	if 'part1' in parts_to_run:
		run_part1(trials['part1'])
	if 'part2' in parts_to_run:
		run_part('part2', trials['part2'])
	if 'part3' in parts_to_run:
		run_part('part3', trials['part3'])

	# Final summary
	n_correct = sum(1 for r in results if r['correct'])
	print('\nExperiment complete. Total correct:', n_correct, 'out of', len(results))
	print('Results saved to:', datafileName+'.tsv')
	dataFile.close()
	myWin.close()
