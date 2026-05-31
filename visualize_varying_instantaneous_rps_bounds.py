"""
Plot instantaneous rps for the varying-speed condition.

This generates four graphs:
- Near displaced: closest point to fixation
- Near displaced: furthest point from fixation
- Far displaced: closest point to fixation
- Far displaced: furthest point from fixation

The x-axis is the base rps value and the y-axis is the resulting instantaneous rps
from the log-polar varying-speed rule used in MOT_pilot_comprehensive.py.
"""

import numpy as np
import matplotlib.pyplot as plt

RADIUS = 6.0
CONDITIONS = {
    'Near displaced': 5.0,
    'Far displaced': 10.0,
}

# Base speed range shown on the x-axis.
# This spans the staircase display range used by the experiment and keeps the plot readable.
BASE_RPS = np.linspace(0.5, 2.2, 300)


def instantaneous_rps(base_rps, l, r, rho):
    """Instantaneous rps for the varying-speed rule."""
    return base_rps * (rho / r)


def closest_and_furthest_rho(l, r):
    """Return the minimum and maximum fixation-to-object distance for a circular path."""
    closest = abs(r - l)
    furthest = r + l
    return closest, furthest


fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
fig.suptitle('Varying-Speed Condition: Instantaneous rps vs Base rps', fontsize=16, fontweight='bold')

panel_specs = [
    ('Near displaced', 'closest', axes[0, 0]),
    ('Near displaced', 'furthest', axes[0, 1]),
    ('Far displaced', 'closest', axes[1, 0]),
    ('Far displaced', 'furthest', axes[1, 1]),
]

for condition_name, distance_type, ax in panel_specs:
    l = CONDITIONS[condition_name]
    closest_rho, furthest_rho = closest_and_furthest_rho(l, RADIUS)

    if distance_type == 'closest':
        rho = closest_rho
        label = f'Closest point to fixation (rho = {rho:.1f} deg)'
        color = 'royalblue'
    else:
        rho = furthest_rho
        label = f'Furthest point from fixation (rho = {rho:.1f} deg)'
        color = 'crimson'

    y = instantaneous_rps(BASE_RPS, l, RADIUS, rho)
    ax.plot(BASE_RPS, y, color=color, linewidth=2.5)
    ax.fill_between(BASE_RPS, y, alpha=0.18, color=color)
    ax.set_title(f'{condition_name} - {distance_type.capitalize()}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Base rps')
    ax.set_ylabel('Instantaneous rps')
    ax.grid(True, alpha=0.3)

    # Add a subtle reference line through the origin at the minimum base speed shown.
    ax.text(
        0.02,
        0.95,
        label,
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='none'),
    )

    # Annotate the scaling factor so the relationship is explicit.
    scale = rho / RADIUS
    ax.text(
        0.02,
        0.82,
        f'y = base rps × {scale:.3f}',
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=9,
        color='dimgray',
    )

for ax in axes.flat:
    ax.set_xlim(BASE_RPS.min(), BASE_RPS.max())

plt.tight_layout(rect=[0, 0, 1, 0.95])
out_file = 'varying_instantaneous_rps_bounds.png'
plt.savefig(out_file, dpi=180, bbox_inches='tight')
print(f'✓ Graph saved to: {out_file}')
