#!/usr/bin/env python3
"""
Simulate a full automated session for MOT_pilot_comprehensive using the
same staircase configuration. Runs headless (no PsychoPy windows) and prints
summary statistics and staircase traces.
"""
import random
import math
from collections import deque

try:
    from psychopy import data
except Exception as e:
    raise RuntimeError('Psychopy is required to run this simulation. Activate psychopy_x86.')

# Parameters (match MOT_pilot_comprehensive.py)
refreshRate = 100.0
stair_nUp = 3
stair_nDown = 1
stair_stepSizes = [.3, .3, .2, .1, .1, .05]
stair_min = 0.7
stair_max = 2.2
stair_trials_per_staircase = 25
stair_start_speed_by_index = {1: 0.7, 2: 1.8}
attention_check_trials_per_condition = 5
attention_check_speed = 0.2

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

# Mapping functions

def stair_value_to_speed(stair_value):
    return stair_min + stair_max - stair_value

def display_speed_to_stair_value(display_speed):
    return stair_min + stair_max - display_speed

# Build attention checks
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

# Build interleaved staircase trials
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
            'queues': {1: deque(), 2: deque()},
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
        insert_after_slots = early_buffer + __import__('numpy').linspace(1, spaced_region, len(attention_checks), dtype=int)
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

# Build staircases using psychopy.data.StairHandler

def build_speed_staircases(trial_specs_all):
    staircases = {}
    for spec in trial_specs_all:
        for staircase_index in (1, 2):
            stair_start_speed = stair_start_speed_by_index[staircase_index]
            staircases[f"{spec['stairKey']}|stair{staircase_index}"] = data.StairHandler(
                startVal=display_speed_to_stair_value(stair_start_speed),
                stepType='lin',
                stepSizes=stair_stepSizes,
                minVal=stair_min,
                maxVal=stair_max,
                nUp=stair_nUp,
                nDown=stair_nDown,
                nTrials=stair_trials_per_staircase,
            )
    return staircases

# Participant model: tracking ability centered at 2.0 rps with steep dropoff above 2.0
# Probability correct = 1 / (1 + exp((speed - 2.0) * slope))

def p_correct_model(speed, slope=5.0):
    return 1.0 / (1.0 + math.exp((speed - 2.0) * slope))

# Run simulation
random.seed(42)

trials_part1 = build_interleaved_stair_trials('Part1', PART1_TRIAL_SPECS)
trials_part2 = build_interleaved_stair_trials('Part2', PART2_TRIAL_SPECS)

speed_staircases = build_speed_staircases(PART1_TRIAL_SPECS + PART2_TRIAL_SPECS)

results = []

for part_name, trial_list in [('Part1', trials_part1), ('Part2', trials_part2)]:
    print(f"Simulating {part_name} ({len(trial_list)} trials)")
    for ti, thisTrial in enumerate(trial_list, 1):
        trial_kind = thisTrial.get('trialKind', 'staircase')
        if trial_kind == 'attention_check':
            speed = attention_check_speed
            p_corr = p_correct_model(speed)
            correct = random.random() < p_corr
        else:
            stair_key = thisTrial['stairKey']
            stair_value = speed_staircases[stair_key].next()
            speed = stair_value_to_speed(stair_value)
            p_corr = p_correct_model(speed)
            correct = random.random() < p_corr
            # apply response to staircase
            speed_staircases[stair_key].addResponse(bool(correct))

        results.append({'part': part_name, 'condition': thisTrial.get('condition'), 'speed': speed, 'correct': correct})

# Summarize
print('\nSimulation summary:')
for part in ('Part1', 'Part2'):
    part_res = [r for r in results if r['part'] == part]
    n = len(part_res)
    n_correct = sum(1 for r in part_res if r['correct'])
    mean_speed = sum(r['speed'] for r in part_res) / n
    print(f"{part}: {n_correct}/{n} correct (mean speed {mean_speed:.3f} rps)")

# Per-staircase final state
print('\nPer-staircase final display speeds (last value):')
for key, sh in speed_staircases.items():
    try:
        last_val = sh._lastVal if hasattr(sh, '_lastVal') else None
        cur = sh._nextVal if hasattr(sh, '_nextVal') else None
    except Exception:
        last_val = None
        cur = None
    # get current displayed speed from sh._nextVal via sh.next() without advancing: call sh.current() not available; use lastVal printed earlier
    # We'll sample by calling sh.next() and then undo by setting internal index back isn't straightforward; instead show reversal count and last stepSizes index
    reversals = len(getattr(sh, 'reversalPoints', []))
    print(f"{key}: reversals={reversals}")

print('\nDone.')
