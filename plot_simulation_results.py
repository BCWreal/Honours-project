#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read simulated session TSV
df = pd.read_csv('simulated_session.tsv', sep='\t')

# Save CSV
df.to_csv('simulated_session.csv', index=False)

# Prepare plot: display speed over trial for each staircase (only staircase trials)
df_stair = df[df['trialKind'] == 'staircase'].copy()

# Identify staircase keys by grouping consecutive trials by condition; but we don't have stairKey column
# We'll group by condition and track appearance order per condition and staircase index via occurrence count

# Build per-condition, per-appearance counts to split into stair1/stair2 alternating within condition.
cond_counters = {}
stair_series = {}
trial_nums = df_stair['trial'].values

for idx, row in df_stair.iterrows():
    cond = row['condition']
    trial_no = row['trial']
    speed = row['speed']
    if cond not in cond_counters:
        cond_counters[cond] = {'count': 0}
    cond_counters[cond]['count'] += 1
    appearance = cond_counters[cond]['count']
    stair_idx = 1 if appearance % 2 == 1 else 2
    key = f"{cond}|stair{stair_idx}"
    if key not in stair_series:
        stair_series[key] = {'trials': [], 'speeds': []}
    stair_series[key]['trials'].append(trial_no)
    stair_series[key]['speeds'].append(speed)

# Plot
plt.figure(figsize=(14, 8))
colors = plt.cm.tab20(np.linspace(0, 1, len(stair_series)))
for (key, data), c in zip(stair_series.items(), colors):
    plt.plot(data['trials'], data['speeds'], marker='o', linestyle='-', label=key, color=c)

plt.xlabel('Trial number')
plt.ylabel('Display speed (rps)')
plt.title('Simulated Staircase Traces — Display speed per trial')
plt.legend(ncol=2, fontsize='small')
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('staircase_traces.png', dpi=150)
print('Saved staircase_traces.png and simulated_session.csv')
