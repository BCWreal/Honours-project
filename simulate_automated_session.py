#!/usr/bin/env python
"""Simulate an automated session using MOT_pilot_comprehensive's staircase logic.

- Uses the same interleaved trial order and PsychoPy StairHandler parameters.
- Simulates a participant with an estimated tracking threshold ~2.0 rps:
  high accuracy below ~1.8 rps, falling accuracy above that point.

Run with the psychopy environment active:

conda activate psychopy_x86
python simulate_automated_session.py
"""

import random
import statistics
from collections import defaultdict

import MOT_pilot_comprehensive as pilot

random.seed(12345)

# Build trials and staircases using the experiment code
trials = pilot.build_session()
# Flatten parts into a single session order: part1 then part2
session_trials = trials['part1'] + trials['part2']

speed_staircases = pilot.build_speed_staircases()

# Participant model: high accuracy up to ~1.8 rps, then accuracy drops toward 0.3 at 2.2 rps
# p_correct(speed): piecewise linear

def p_correct_for_speed(speed):
    if speed <= 1.8:
        return 0.95
    # linear decay from 1.8->2.2: 0.95 -> 0.30
    slope = (0.30 - 0.95) / (2.2 - 1.8)
    p = 0.95 + slope * (speed - 1.8)
    return max(0.05, min(0.95, p))

results = []
per_stair_stats = defaultdict(list)

trial_index = 0
for t in session_trials:
    trial_index += 1
    if t.get('trialKind') == 'attention_check':
        # attention trials at 0.2 rps: assume participant nearly always correct
        speed = t['speed']
        p = 0.99
        resp = random.random() < p
        correct = bool(resp)
        stair_value = None
        stair_key = None
    else:
        stair_key = t['stairKey']
        stair_value = speed_staircases[stair_key].next()
        speed = pilot.stair_value_to_speed(stair_value)
        p = p_correct_for_speed(speed)
        resp = random.random() < p
        correct = bool(resp)
        # update staircase
        speed_staircases[stair_key].addResponse(correct)
        per_stair_stats[stair_key].append({'trial': trial_index, 'speed': speed, 'correct': correct})

    results.append({'trial': trial_index, 'part': t['part'], 'condition': t.get('condition'), 'stairKey': t.get('stairKey'), 'trialKind': t.get('trialKind', 'staircase'), 'speed': speed, 'correct': correct, 'p_model': p if t.get('trialKind')!='attention_check' else 0.99})

# Summaries
print('\nSimulated automated session summary')
print('Total trials:', len(results))
print('Overall accuracy: {0:.3f}'.format(sum(1 for r in results if r['correct']) / len(results)))

# Per-staircase summaries
print('\nPer-staircase summaries:')
for stair_key, trials_list in per_stair_stats.items():
    acc = sum(1 for x in trials_list if x['correct']) / len(trials_list)
    mean_speed = statistics.mean(x['speed'] for x in trials_list)
    last_speed = trials_list[-1]['speed']
    print(f"{stair_key}: trials={len(trials_list)} mean_speed={mean_speed:.3f} last_speed={last_speed:.3f} acc={acc:.3f}")

# Print first 40 trial outcomes for inspection
print('\nFirst 40 simulated trials:')
print('trial\tpart\tcond\ttype\tspeed\tp_model\tcorrect')
for r in results[:40]:
    print(f"{r['trial']}\t{r['part']}\t{r['condition']}\t{r['trialKind']}\t{r['speed']:.3f}\t{r['p_model']:.3f}\t{int(r['correct'])}")

# Save to file
outf = 'simulated_session.tsv'
with open(outf, 'w') as f:
    f.write('trial\tpart\tcondition\ttrialKind\tspeed\tp_model\tcorrect\n')
    for r in results:
        f.write(f"{r['trial']}\t{r['part']}\t{r['condition']}\t{r['trialKind']}\t{(r['speed'] if r['speed'] is not None else -999):.3f}\t{r['p_model']:.3f}\t{int(r['correct'])}\n")

print(f"\nSaved simulated session to: {outf}\n")
