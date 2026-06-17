# Rapid Reaching Robot Arm — CS403 Group Project

A controller that drives a 6-DoF robotic arm to rapidly reach a set of randomly generated 3D target points in the **MuJoCo** physics simulator. Built for COMPSCI 403 (Introduction to Robotics) at UMass Amherst, where it placed **2nd of 7 teams** in the class competition with a mean completion time of **3.92 seconds**.

**Authors:** Cadence Young, Marwan Hegab, Caroline Zouloumian, Madeline Gelnett

## Approach

At each control step the controller:

1. **Target selection** — picks the nearest unreached point by Euclidean distance (with an optional Nearest-Neighbor TSP pre-planning of the full route).
2. **Inverse kinematics** — computes the Cartesian position error and solves for joint updates using **Damped Least Squares (DLS)**, which stays stable near kinematic singularities.
3. **Joint-space PD control** — converts desired joint positions into torques with a per-joint PD controller, adding MuJoCo's bias term to compensate for gravity, Coriolis, and centrifugal forces.
4. **Torque clipping** — clamps torques to each actuator's control range as a safety measure.

The full method, experiments, challenges, and results are documented in **[`paper.pdf`](paper.pdf)**.

```
Δq = α · Jᵀ (J Jᵀ + λ²I)⁻¹ (x_goal − x_ee)      # DLS inverse kinematics
τ  = Kp (q_des − q) − Kd q̇ + τ_bias              # PD control w/ gravity compensation
```

## Repository structure

```
.
├── YourControlCode.py        # the controller (target planning, IK, PD control)
├── Run_RapidReachingEnv.py   # simulation environment / competition harness
├── PointManager.py           # target point generation & bookkeeping
├── Robot/                     # MuJoCo model
│   ├── miniArm.xml
│   ├── miniArm_with_points.xml
│   └── meshes/                # STL/OBJ meshes for the arm
└── paper.pdf                  # project report
```

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the rapid-reaching simulation
python Run_RapidReachingEnv.py
```

> **Note:** MuJoCo requires a native build of Python matching your CPU architecture (e.g. arm64 on Apple Silicon).

## License

Released under the [MIT License](LICENSE).
