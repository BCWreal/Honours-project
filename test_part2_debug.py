#!/usr/bin/env python
"""
Minimal test to verify Part 2 code path is being triggered
"""
import sys
import os

# Set environment
os.environ['MOT_AUTO_ADVANCE'] = '1'
os.environ['MOT_PARTS'] = 'part2'

# Verify the key condition for Part 2 log-eccentricity
part_name = 'part2'
basicShape = 'circle'

# Test different offset values
test_cases = [
    (0.0, 0.0, "centred (should use constant speed)"),
    (5.0, 0.0, "near_displaced (should use log-eccentricity)"),
    (10.0, 0.0, "far_displaced (should use log-eccentricity)"),
]

print("=" * 70)
print("Testing Part 2 code path condition:")
print(f"  part_name = '{part_name}'")
print(f"  basicShape = '{basicShape}'")
print("=" * 70)

for cx, cy, description in test_cases:
    condition_result = (part_name == 'part2' and basicShape == 'circle' and (cx != 0.0 or cy != 0.0))
    print(f"\n{description}")
    print(f"  cx={cx}, cy={cy}")
    print(f"  Condition result: {condition_result}")
    if condition_result:
        print(f"  ✓ LOG-ECCENTRICITY equations WILL be used")
    else:
        print(f"  ✗ Falling back to CONSTANT SPEED")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("=" * 70)
print("""
If you see:
  - "LOG-ECCENTRICITY equations WILL be used" for near_displaced and far_displaced
  - "Falling back to CONSTANT SPEED" for centred
Then the code paths are correct.

If centred also says LOG-ECCENTRICITY, then we have an issue.
If near/far_displaced say CONSTANT SPEED, then the condition is preventing
the log-eccentricity code from running!
""")
