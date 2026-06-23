#!/usr/bin/env python
"""
Test staircase stepping behavior with the fixed nUp/nDown parameters.
"""

from psychopy import data

# Staircase parameters (matching MOT_pilot_comprehensive.py)
stair_nUp = 3
stair_nDown = 1
stair_stepSizes = [.3, .3, .2, .1, .1, .05]
stair_min = 0.7
stair_max = 2.2

def stair_value_to_speed(stair_value):
    """Map the StairHandler value so speed increases when StairHandler steps down."""
    return stair_min + stair_max - stair_value

def display_speed_to_stair_value(display_speed):
    """Convert a displayed speed back into the StairHandler value space."""
    return stair_min + stair_max - display_speed

# Create staircase 1 (starts at 0.7 rps)
stair_start_speed_1 = 0.7
stair_value_start = display_speed_to_stair_value(stair_start_speed_1)

print(f"Staircase 1 test (starting speed = {stair_start_speed_1} rps)")
print(f"Starting stair_value = {stair_value_start}")
print(f"Bounds: stair_min = {stair_min}, stair_max = {stair_max}")
print()

staircase1 = data.StairHandler(
    startVal=stair_value_start,
    stepType='lin',
    stepSizes=stair_stepSizes,
    minVal=stair_min,
    maxVal=stair_max,
    nUp=stair_nUp,
    nDown=stair_nDown,
    nTrials=25,
    extraInfo={'staircase': 1}
)

# Simulate a realistic response sequence
# Scenario: mostly correct with occasional errors
responses = [
    True,    # T1: correct (1 correct)
    True,    # T2: correct (2 correct)
    True,    # T3: correct (3 correct) → should step UP (harder, speed increases)
    True,    # T4: correct (1 correct)
    False,   # T5: incorrect → should step DOWN (easier, speed decreases) + reversal
    True,    # T6: correct (1 correct)
    True,    # T7: correct (2 correct)
    True,    # T8: correct (3 correct) → should step UP (harder, speed increases)
    False,   # T9: incorrect → should step DOWN (easier, speed decreases) + reversal
    True,    # T10: correct (1 correct)
]

print("Trial | Response | Speeds | StairValue | DisplaySpeed | Reversals")
print("------|----------|--------|------------|--------------|----------")

for i, resp in enumerate(responses, 1):
    # Get current speed before response
    current_stair_value = staircase1.next()
    current_speed = stair_value_to_speed(current_stair_value)
    
    # Add response (this will step if needed)
    staircase1.addResponse(resp)
    
    # Get reversal count
    n_reversals = len(staircase1.reversalPoints) if hasattr(staircase1, 'reversalPoints') else 0
    
    print(f"  {i:2d}  |   {str(resp):5s}  | {current_speed:.2f}  |    {current_stair_value:.4f}  |    {current_speed:.4f}   |     {n_reversals}")

print()
print("Expected behavior:")
print("- T1-T3: 3 correct responses → should trigger step UP (speed increases)")
print("- T5: 1 incorrect → should trigger step DOWN (speed decreases), also reversal")
print("- T6-T8: 3 correct responses → should trigger step UP (speed increases)")
print("- T9: 1 incorrect → should trigger step DOWN (speed decreases), also reversal")
print()
print(f"Final staircase value: {staircase1.current()}")
print(f"Final display speed: {stair_value_to_speed(staircase1.current()):.4f} rps")
