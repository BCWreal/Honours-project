"""
MOT_pilot_debug_varying_speed_comprehensive.py

DEBUG ISOLATED VERSION: Part 2 Only - Varying Speed Condition
COMPREHENSIVE VERSION with detailed results logging matching MOT Circular.py

This file contains ONLY the code necessary to run Part 2 (Varying Speed) from MOT_pilot.
Part 2 tests circular trajectories with three fixation offset conditions using log-polar 
speed equations for displaced conditions to maintain constant revolution time.

Results logging includes: trial details, initial angles, reversal times, trial duration

CRITICAL PART 2 FEATURES:
 - 2 objects per trial (1 target, 1 distractor)
 - Circular trajectories with radius 6 deg
 - 3 conditions: centred (0°), near_displaced (5°), far_displaced (10°)
 - Speeds: [1.0,1.5,1.7,1.9,2.0,2.2] rps, 4 repeats each = 24 trials/condition
 - Total: 72 trials for Part 2
 - LOG-POLAR MOTION: For displaced conditions, speed is modulated using log-polar 
   speed equation to maintain correct average revolution time
 - Centred condition uses constant angular speed (no log-polar modulation)
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
repeats_per_speed = 4

# Sounds
beep_correct = sound.Sound(value='C', secs=0.08)  # simple tone
beep_incorrect = sound.Sound(value='A', secs=0.12)

timeTillReversalMin = 0.5
timeTillReversalMax = 2.0
badTimingCushion = 0.3
part2_log_polar_V_base = 1.0  # Base speed in log-polar space; scaled by trial speed to match nominal rps
part2_log_polar_ref_speed = 1.0  # Reference speed (rps) for which V_base was calibrated


def get_reversal_times(trial_duration_sec):
	"""Generate random reversal times during trial"""
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
	CRITICAL FOR PART 2: Compute the mean of 1/rho over a full circle.
	Used to scale V for log-polar motion to ensure correct average revolution time.
	
	For log-polar speed equation phi_dot = (V/r) * rho, the time for one full revolution is:
	T = (r/V) * ∫[0, 2π] (1/rho(φ)) dφ = (r/V) * 2π * mean(1/rho)
	
	To get T = 1/speed, we need: V = speed * r * 2π * mean(1/rho)
	
	Parameters:
	- l: distance from fixation to circle centre (eccentricity)
	- r: radius of the circular path
	- num_samples: number of sample points around circle for numerical integration
	
	Returns:
	- mean_inv_rho: average of 1/rho values across the circle
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
	CRITICAL FOR PART 2: Update phi using constant log-polar speed equation.
	
	This implements motion in log-polar coordinates where speed is constant in 
	the log-polar space, resulting in position-dependent speed in Cartesian space.
	
	phi_dot = (V / r) * rho
	where rho = sqrt(l^2 + r^2 + 2*l*r*cos(phi))
	
	This means the object moves faster when farther from fixation and slower when closer,
	maintaining a constant average revolution time despite the position-dependent speed.
	
	Parameters:
	- phi: current angle around circle centre
	- V: desired constant speed in log-polar space from fixation
	- l: distance from fixation to circle centre (eccentricity)
	- r: radius of the circular path
	- dt: time step (1/refreshRate)
	- direction: +1 or -1 for direction of motion (direction reversals)
	
	Returns:
	- phi_new: updated angle after time step dt
	"""
	rho = np.sqrt(l*l + r*r + 2.0*l*r*np.cos(phi))
	phi_dot = (V / r) * rho
	phi_dot = phi_dot * direction  # apply reversal direction
	phi_new = phi + phi_dot * dt
	phi_new = phi_new % (2 * np.pi)
	return phi_new


def get_xy_from_phi(phi, offset_xy, l, r):
	"""Convert log-polar coordinates to Cartesian"""
	x = offset_xy[0] + r * cos(phi)
	y = offset_xy[1] + r * sin(phi)
	return x, y


# -------------------- Trajectory generators --------------------
def circle_xy(radius_deg, angle_rad, timeSeconds=None, speed=None, numRing=0):
	"""
	Convert angle to x,y position on circle (with optional temporal modulation).
	Part 2 uses this for all motion calculations.
	"""
	r = radius_deg
	if timeSeconds is not None and speed is not None:
		rThis = r + waveForm('sin', speed, timeSeconds, numRing) * r * ampTemporalRadiusModulation
		rThis += r * RFcontourAmp * RFcontourCalcModulation(angle_rad, RFcontourFreq, RFcontourPhase)
	else:
		rThis = r
	return rThis * cos(angle_rad), rThis * sin(angle_rad)


def draw_trajectory(myWin, basicShape, radius_deg, cx, cy, num_points=120):
	"""
	Draw the actual trajectory (white line) that objects traveled during the trial.
	"""
	trajectory_points = []
	for i in range(num_points):
		angle_rad = 2 * np.pi * i / num_points
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
	"""Generate trial sequence for Part 2 only"""
	trials = []
	for cond in conditions:
		for sp in speeds_base:
			for rep in range(repeats_per_speed):
				trials.append({'part': part_name, 'condition': cond, 'speed': sp})
	random.shuffle(trials)
	return trials


def build_session_part2_only():
	"""Build only Part 2 trials for debugging"""
	fixation_conditions = ['centred', 'near_displaced', 'far_displaced']
	part2_trials = make_trials_for_part('Part2_VaryingSpeed', fixation_conditions)
	return {'part2': part2_trials}


# -------------------- Minimal run loop for verification --------------------
def run_checks_and_report(trials_by_part):
	"""Verify Part 2 trial setup"""
	circle_radius_ok = (abs(radii[0] - 6.0) < 1e-6)

	counts = {k: len(v) for k, v in trials_by_part.items()}

	speed_counts = {}
	for pname, tlist in trials_by_part.items():
		sc = {}
		for t in tlist:
			sc[t['speed']] = sc.get(t['speed'], 0) + 1
		speed_counts[pname] = sc

	print('\n=== MOT_pilot PART 2 DEBUG verification report ===')
	print('Circular radii set to 6 deg:', circle_radius_ok)
	for pname in counts:
		print(f"{pname}: {counts[pname]} trials (expected 72)")
		print('  speed distribution:', speed_counts[pname])

	total_trials = sum(counts.values())
	print('Total trials for Part 2 =', total_trials, '(expected 72)')

	print('\nFeedback sounds available: correct/inaccurate assigned')
	print('beep_correct:', beep_correct)
	print('beep_incorrect:', beep_incorrect)

	issues = []
	for pname in counts:
		if counts[pname] != 72:
			issues.append(f"{pname} has {counts[pname]} trials (expected 72)")
	
	if total_trials != 72:
		issues.append('Total trials mismatch for Part 2')

	if not circle_radius_ok:
		issues.append('Circular radius not 6 deg')

	if issues:
		print('\nIssues detected:')
		for it in issues:
			print('-', it)
	else:
		print('\nNo issues detected. Ready to run Part 2 debugging.')


if __name__ == '__main__':
	random.seed(int(time.time()))
	trials = build_session_part2_only()
	run_checks_and_report(trials)

	# -------------------- Data file setup --------------------
	subject = 'temp'
	session = 1
	timeAndDateStr = time.strftime("%d%b%Y_%H-%M", time.localtime())
	if os.path.isdir('.'+os.sep+'dataRaw'):
		dataDir='dataRaw'
	else:
		print('"dataRaw" directory does not exist, so saving data in present working directory')
		dataDir='.'
	datafileName = dataDir+'/'+subject+ '_' + str(session) + '_MOT_pilot_part2_debug_'+timeAndDateStr
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
		"""
		PART 2 MAIN TRIAL LOOP
		
		Key behaviors specific to Part 2:
		1. CENTRED condition (cx=0, cy=0): Uses constant angular speed
		2. DISPLACED conditions (cx=5 or 10, cy=0): Uses LOG-POLAR speed equation
		   - Computes V_scaled based on mean_inv_rho for the eccentricity
		   - Each object moves independently with phi_dot = (V/r) * rho
		   - This maintains constant average revolution time despite position-dependent speed
		"""
		print(f"Running {part_name} with {len(trial_list)} trials")
		for ti, thisTrial in enumerate(trial_list):
			# PART 2: Always centred and circular shape (fixation is in center, trajectory can be offset)
			cond = thisTrial['condition']
			
			# Determine trajectory center based on condition
			if cond == 'centred':
				cx = practice_trajectoryCenterXDeg[0]  # 0 deg
			elif cond == 'near_displaced':
				cx = practice_trajectoryCenterXDeg[1]  # 5 deg
			elif cond == 'far_displaced':
				cx = practice_trajectoryCenterXDeg[2]  # 10 deg
			cy = 0.0
			basicShape = 'circle'  # Part 2 always uses circles

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
				
				# ==================== CRITICAL PART 2 LOGIC ====================
				if part_name == 'part2' and basicShape == 'circle' and (cx != 0.0 or cy != 0.0):
					# DISPLACED CONDITION: Use log-polar speed equation
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
					# CENTRED CONDITION: Both objects move together at the same constant angular speed
					if frameN == 0 and part_name == 'part2':
						print(f"DEBUG: Part 2 CENTRED condition (constant speed). part_name={part_name}, basicShape={basicShape}, cx={cx}, cy={cy}")
					
					angleStep = direction * speed * 2 * pi / refreshRate
					currAngle = (currAngle + angleStep) % (2 * pi)
					otherAngle = (otherAngle + angleStep) % (2 * pi)

					# compute positions using circle equation
					x1, y1 = circle_xy(radii[0], currAngle, timeSec, speed, 0)
					x2, y2 = circle_xy(radii[0], otherAngle, timeSec, speed, 0)

				# Handle direction reversals
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

	# ==================== RUN PART 2 ONLY ====================
	print('\n--- Starting Part 2: Varying Speed (DEBUG MODE) ---')
	run_part('part2', trials['part2'])

	# Final summary
	n_correct = sum(1 for r in results if r['correct'])
	print('\nPart 2 complete. Total correct:', n_correct, 'out of', len(results))
	print('Results saved to:', datafileName+'.tsv')
	dataFile.close()
	myWin.close()
