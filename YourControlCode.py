import mujoco
import numpy as np

class YourCtrl:
    # --- Control Gains (Now as NumPy Arrays for per-joint tuning) ---
    # Gains are higher for joints closer to the base (shoulder, elbow)
    # and lower for joints closer to the end-effector (wrist).
    Kp = np.array([400.0, 800.0, 1200.0, 2200.0, 100.0, 100.0])
    # Damping is calculated element-wise based on the new Kp array.
    Kd = 2 * np.sqrt(0.9 * Kp)

    # --- IK Parameters ---
    alpha = 0.98 # IK step fraction (- started at 0.5)
    lambda_dls = 0.09 # DLS damping (- started at 0.05)
    reach_thresh = 0.01 # 1 cm

    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, target_points):
        self.m, self.d = m, d
        
        self.targets = target_points.copy() # (3, N) - This will be reordered
        self.N = self.targets.shape[1]
        
        # --- Path Planning State ---
        self.current_target_idx = 0

        self.ee_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "EE_Frame") # end effector body id
        self.nu = m.nu

        # --- Jacobian Buffers ---
        self.jacp = np.zeros((3, m.nv)) # position
        self.jacr = np.zeros((3, m.nv)) # rotation

        self.q_des = self.d.qpos[:self.nu].copy() # desired joint angles

        # Pre-plan the entire path at initialization
        self._plan_path_tsp()

    def _plan_path_tsp(self):
        """
        Calculates an efficient path through all target points using a
        Nearest Neighbor heuristic for the Traveling Salesman Problem (TSP).
        This pre-plans the entire route at the start, which is often more
        efficient than a simple greedy approach. The 'self.targets'
        array is reordered according to this planned path.
        """
        points = self.targets.T.tolist()
        ee_start_pos = self.d.xpos[self.ee_bid]
        start_idx = np.linalg.norm(self.targets.T - ee_start_pos, axis=1).argmin()
        
        path = [points.pop(start_idx)]
        ordered_targets = [self.targets[:, start_idx]]

        while points:
            last_point = path[-1]
            nearest_idx = np.linalg.norm(np.array(points) - last_point, axis=1).argmin()
            next_point = points.pop(nearest_idx)
            path.append(next_point)
            
            original_idx = np.where((self.targets.T == next_point).all(axis=1))[0][0]
            ordered_targets.append(self.targets[:, original_idx])
            
        self.targets = np.array(ordered_targets).T
        print("TSP path planned. New target order generated.")


    def _update_target(self):
        """
        Checks if the current target in the pre-planned path is reached
        and updates the target index to the next point in the sequence.
        """
        if self.current_target_idx >= self.N:
            return None # Finished

        current_target_pos = self.targets[:, self.current_target_idx]
        ee_pos = self.d.xpos[self.ee_bid]
        
        distance = np.linalg.norm(ee_pos - current_target_pos)

        if distance < self.reach_thresh:
            print(f"Reached target {self.current_target_idx + 1}/{self.N}. Moving to next.")
            self.current_target_idx += 1
            if self.current_target_idx >= self.N:
                return None

        return self.current_target_idx


    def _ik_step(self, goal):
        ee = self.d.xpos[self.ee_bid] # Current end effector position
        err = goal - ee # Vector position error
        mujoco.mj_jac(self.m, self.d, self.jacp, self.jacr, ee, self.ee_bid) # Compute Jacobian
        Jp = self.jacp # Position Jacobian
        J = Jp
        JJt = J @ J.T + (self.lambda_dls ** 2) * np.eye(3) 
        dq = self.alpha * (J.T @ np.linalg.solve(JJt, err)) # change in q
        self.q_des = self.d.qpos[:self.nu] + dq # q_{k+1} = q_k + delta*q

    def CtrlUpdate(self):
        # Check if the current target is reached and get the next one
        idx = self._update_target()
        if idx is None:
            return np.zeros(self.nu)  # finished

        self._ik_step(self.targets[:, idx]) # perform IK step

        q = self.d.qpos[:self.nu] # joint positions
        qd = self.d.qvel[:self.nu] # joint velocities
        
        # Calculate PD torque, now using Kp and Kd arrays for per-joint control
        tau = self.Kp * (self.q_des - q) - self.Kd * qd + self.d.qfrc_bias[:self.nu] # joint torque bias

        for k in range(self.nu): # iterate over actuators
            lo, hi = self.m.actuator_ctrlrange[k] # actuator limits
            tau[k] = np.clip(tau[k], lo, hi) # clip to limits

        return tau