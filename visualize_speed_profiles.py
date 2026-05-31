"""
Visualization of tangential speed profiles and instantaneous rps for all MOT conditions.

Creates 12 graphs total:
- 6 graphs of tangential speed vs position in circle
- 6 graphs of instantaneous rps vs time
- 3 speeds: 1.0 rps, 1.5 rps, 2.0 rps
- Part 1 & Part 2 conditions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from math import pi, cos, sin

# ==================== Motion Rule Functions ====================

def rho_from_phi(phi, l, r):
    """Distance from fixation to object at angle phi (log-polar)."""
    return np.sqrt(l * l + r * r + 2.0 * l * r * np.cos(phi))

def phi_dot_log_eccentricity_base_speed(phi, base_rps, l, r):
    """Varying-speed rule: angular velocity based on distance from fixation."""
    rho = rho_from_phi(phi, l, r)
    return 2 * np.pi * base_rps * (rho / r)

def circle_xy(radius_deg, angle_rad):
    """Standard circle trajectory."""
    return radius_deg * np.cos(angle_rad), radius_deg * np.sin(angle_rad)

def sine_circle_xy(radius_deg, angle_rad):
    """Sine-wave modulated circle."""
    sine_amplitude_deg = 0.7
    sine_lobes = 12
    r = radius_deg + sine_amplitude_deg * np.sin(sine_lobes * angle_rad)
    return r * np.cos(angle_rad), r * np.sin(angle_rad)

def ellipse_xy(circle_radius, angle_rad):
    """Ellipse trajectory."""
    aspect_ratio = 1.6
    rotation_rad = pi / 4.0
    
    def ellipse_circumference_approx(a, b):
        h = ((a - b) ** 2) / ((a + b) ** 2)
        return pi * (a + b) * (1 + ((3 * h) / (10 + np.sqrt(4 - 3 * h))))
    
    def ellipse_axes_matching_circumference(radius, aspect_ratio):
        a_raw = radius * aspect_ratio
        b_raw = radius / aspect_ratio
        target_circ = 2 * pi * radius
        raw_circ = ellipse_circumference_approx(a_raw, b_raw)
        if raw_circ <= 0:
            return radius, radius
        scale = target_circ / raw_circ
        return a_raw * scale, b_raw * scale
    
    a, b = ellipse_axes_matching_circumference(circle_radius, aspect_ratio)
    x_unrot = a * np.cos(angle_rad)
    y_unrot = b * np.sin(angle_rad)
    
    # Rotate
    c = np.cos(rotation_rad)
    s = np.sin(rotation_rad)
    x = x_unrot * c - y_unrot * s
    y = x_unrot * s + y_unrot * c
    return x, y

def diamond_xy(radius_deg, angle_rad):
    """Diamond trajectory."""
    p = (angle_rad % (2 * pi)) / (2 * pi)
    half_side = 1.0 / np.sqrt(2.0)
    u = p * 4.0
    
    if u < 1.0:
        x = half_side
        y = -half_side + (u * 2.0 * half_side)
    elif u < 2.0:
        x = half_side - ((u - 1.0) * 2.0 * half_side)
        y = half_side
    elif u < 3.0:
        x = -half_side
        y = half_side - ((u - 2.0) * 2.0 * half_side)
    else:
        x = -half_side + ((u - 3.0) * 2.0 * half_side)
        y = -half_side
    
    # Rotate 45 degrees
    c = np.cos(pi / 4.0)
    s = np.sin(pi / 4.0)
    x_rot = x * c - y * s
    y_rot = x * s + y * c
    return radius_deg * x_rot, radius_deg * y_rot

# ==================== Trajectory Generators ====================

def generate_trajectory(condition, motion_rule, shape, base_rps, num_samples=360, trial_duration=2.0, radius=6.0):
    """
    Generate trajectory and compute tangential speeds.
    
    Returns: angles, xy_positions, distances_from_fixation, instantaneous_rps, time_array
    """
    
    displacement = {'centred': 0.0, 'near_displaced': 2.0, 'far_displaced': 4.0}.get(condition, 0.0)
    
    # For Part 2 shapes
    if shape == 'diamond':
        trajectory_func = lambda angle: diamond_xy(radius, angle)
    elif shape == 'sine_circle':
        trajectory_func = lambda angle: sine_circle_xy(radius, angle)
    elif shape == 'ellipse':
        trajectory_func = lambda angle: ellipse_xy(radius, angle)
    else:  # circle
        trajectory_func = lambda angle: circle_xy(radius, angle)
    
    if motion_rule == 'varying':
        # Varying speed: use log-polar speed equation
        l = displacement  # distance from fixation to circle centre
        r = radius
        
        angles = []
        xy_positions = []
        distances = []
        rps_values = []
        
        dt = 0.001  # time step
        num_steps = int(trial_duration / dt)
        phi = 0.0
        
        for step in range(num_steps):
            angles.append(phi)
            x, y = trajectory_func(phi)
            xy_positions.append((x, y))
            
            # Distance from fixation
            dist = np.sqrt(x**2 + y**2)
            distances.append(dist)
            
            # Instantaneous rps based on angular velocity
            phi_dot = phi_dot_log_eccentricity_base_speed(phi, base_rps, l, r)
            rps = phi_dot / (2 * pi)
            rps_values.append(rps)
            
            # Update angle
            phi += phi_dot * dt
            phi = phi % (2 * pi)
        
        time_array = np.arange(num_steps) * dt
    
    else:
        # Standard speed: constant rps
        l = displacement
        r = radius
        
        angles = []
        xy_positions = []
        distances = []
        rps_values = []
        
        dt = 0.001
        num_steps = int(trial_duration / dt)
        phi = 0.0
        
        for step in range(num_steps):
            angles.append(phi)
            x, y = trajectory_func(phi)
            xy_positions.append((x, y))
            
            dist = np.sqrt(x**2 + y**2)
            distances.append(dist)
            
            rps_values.append(base_rps)
            
            # Constant angular velocity
            phi_dot = 2 * pi * base_rps
            phi += phi_dot * dt
            phi = phi % (2 * pi)
        
        time_array = np.arange(num_steps) * dt
    
    return np.array(angles), np.array(xy_positions), np.array(distances), np.array(rps_values), time_array

def compute_tangential_speed(xy_positions, angles, rps_values):
    """
    Compute tangential speed along trajectory.
    Speed = distance traveled / time elapsed
    """
    speeds = []
    for i in range(len(xy_positions) - 1):
        x1, y1 = xy_positions[i]
        x2, y2 = xy_positions[i + 1]
        dist_traveled = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Time step
        dt = 0.001
        
        # Tangential speed in deg/s
        speed = dist_traveled / dt
        speeds.append(speed)
    
    speeds.append(speeds[-1])  # duplicate last value
    return np.array(speeds)

# ==================== Part 1 Conditions ====================

part1_conditions = [
    ('centred', 'standard', 'circle', 'Centred / Standard'),
    ('near_displaced', 'standard', 'circle', 'Near Displaced / Standard'),
    ('far_displaced', 'standard', 'circle', 'Far Displaced / Standard'),
    ('near_displaced', 'varying', 'circle', 'Near Displaced / Varying'),
    ('far_displaced', 'varying', 'circle', 'Far Displaced / Varying'),
]

# ==================== Part 2 Conditions ====================

part2_conditions = [
    ('diamond', 'shape', 'diamond', 'Diamond'),
    ('sine_circle', 'shape', 'sine_circle', 'Sine Circle'),
    ('ellipse', 'shape', 'ellipse', 'Ellipse'),
]

# ==================== Create Graphs ====================

rps_speeds = [1.0, 1.5, 2.0]
rps_labels = ['1.0 rps', '1.5 rps', '2.0 rps']

# ========== TANGENTIAL SPEED GRAPHS ==========

print("Generating tangential speed profile graphs...")

# Part 1 - Tangential Speed (3 graphs, one per rps)
for rps_idx, (rps, rps_label) in enumerate(zip(rps_speeds, rps_labels)):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'Part 1: Tangential Speed Profiles ({rps_label})', fontsize=16, fontweight='bold')
    
    for idx, (condition, motion_rule, shape, label) in enumerate(part1_conditions):
        ax = plt.subplot(3, 2, idx + 1)
        
        angles, xy_pos, dists, rps_vals, time_arr = generate_trajectory(
            condition, motion_rule, shape, rps
        )
        tangential_speeds = compute_tangential_speed(xy_pos, angles, rps_vals)
        
        # Plot speed vs position along trajectory (mapped to 0-360 degrees)
        angle_degrees = (angles % (2 * pi)) * 180 / pi
        ax.plot(angle_degrees, tangential_speeds, linewidth=2, color='steelblue')
        ax.fill_between(angle_degrees, tangential_speeds, alpha=0.3, color='steelblue')
        
        ax.set_xlabel('Position in Circle (degrees)', fontsize=10)
        ax.set_ylabel('Tangential Speed (deg/s)', fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 360)
        
        # Add mean speed annotation
        mean_speed = np.mean(tangential_speeds)
        ax.text(0.98, 0.97, f'Mean: {mean_speed:.2f} deg/s', 
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'part1_tangential_speed_{rps}rps.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: part1_tangential_speed_{rps}rps.png")

# Part 2 - Tangential Speed (3 graphs, one per rps)
for rps_idx, (rps, rps_label) in enumerate(zip(rps_speeds, rps_labels)):
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f'Part 2: Tangential Speed Profiles ({rps_label})', fontsize=16, fontweight='bold')
    
    for idx, (condition, motion_rule, shape, label) in enumerate(part2_conditions):
        ax = plt.subplot(2, 2, idx + 1)
        
        angles, xy_pos, dists, rps_vals, time_arr = generate_trajectory(
            condition, motion_rule, shape, rps
        )
        tangential_speeds = compute_tangential_speed(xy_pos, angles, rps_vals)
        
        angle_degrees = (angles % (2 * pi)) * 180 / pi
        ax.plot(angle_degrees, tangential_speeds, linewidth=2, color='darkgreen')
        ax.fill_between(angle_degrees, tangential_speeds, alpha=0.3, color='darkgreen')
        
        ax.set_xlabel('Position in Circle (degrees)', fontsize=10)
        ax.set_ylabel('Tangential Speed (deg/s)', fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 360)
        
        mean_speed = np.mean(tangential_speeds)
        ax.text(0.98, 0.97, f'Mean: {mean_speed:.2f} deg/s', 
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'part2_tangential_speed_{rps}rps.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: part2_tangential_speed_{rps}rps.png")

# ========== INSTANTANEOUS RPS GRAPHS ==========

print("\nGenerating instantaneous rps vs time graphs...")

# Part 1 - Instantaneous RPS (3 graphs, one per rps)
for rps_idx, (rps, rps_label) in enumerate(zip(rps_speeds, rps_labels)):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'Part 1: Instantaneous RPS vs Time ({rps_label})', fontsize=16, fontweight='bold')
    
    for idx, (condition, motion_rule, shape, label) in enumerate(part1_conditions):
        ax = plt.subplot(3, 2, idx + 1)
        
        angles, xy_pos, dists, rps_vals, time_arr = generate_trajectory(
            condition, motion_rule, shape, rps, trial_duration=2.0
        )
        
        # Plot only first 2 seconds (trial duration)
        ax.plot(time_arr, rps_vals, linewidth=2, color='crimson')
        ax.fill_between(time_arr, rps_vals, alpha=0.3, color='crimson')
        
        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.set_ylabel('Instantaneous RPS', fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 2.0)
        
        mean_rps = np.mean(rps_vals)
        ax.text(0.98, 0.97, f'Mean: {mean_rps:.3f} rps', 
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'part1_instantaneous_rps_{rps}rps.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: part1_instantaneous_rps_{rps}rps.png")

# Part 2 - Instantaneous RPS (3 graphs, one per rps)
for rps_idx, (rps, rps_label) in enumerate(zip(rps_speeds, rps_labels)):
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(f'Part 2: Instantaneous RPS vs Time ({rps_label})', fontsize=16, fontweight='bold')
    
    for idx, (condition, motion_rule, shape, label) in enumerate(part2_conditions):
        ax = plt.subplot(2, 2, idx + 1)
        
        angles, xy_pos, dists, rps_vals, time_arr = generate_trajectory(
            condition, motion_rule, shape, rps, trial_duration=2.0
        )
        
        ax.plot(time_arr, rps_vals, linewidth=2, color='darkviolet')
        ax.fill_between(time_arr, rps_vals, alpha=0.3, color='darkviolet')
        
        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.set_ylabel('Instantaneous RPS', fontsize=10)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 2.0)
        
        mean_rps = np.mean(rps_vals)
        ax.text(0.98, 0.97, f'Mean: {mean_rps:.3f} rps', 
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'part2_instantaneous_rps_{rps}rps.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: part2_instantaneous_rps_{rps}rps.png")

print("\n✓ All 12 graphs generated successfully!")
print("\nGenerated files:")
print("  - 6 tangential speed graphs (part1/part2 × 3 speeds)")
print("  - 6 instantaneous rps graphs (part1/part2 × 3 speeds)")
