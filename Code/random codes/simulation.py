# quadcopter_flight_sim.py
"""
Quadcopter Flight Dynamics + PID Control Simulation
- Models altitude + roll/pitch attitude dynamics (simplified rigid body)
- Includes ESC calibration errors (motor gain mismatch)
- PID controller for altitude + attitude
- Outputs performance metrics (rise time, overshoot, steady-state error)
- Plots altitude, angles, motor throttles
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------- Simulation Parameters ----------------
dt = 0.002          # time step (s)
T = 15.0            # total time (s)
steps = int(T / dt)
t = np.linspace(0, T, steps)

# Physical quadcopter parameters
mass = 1.2          # kg
g = 9.81            # m/s^2
arm_length = 0.2    # meters
Ixx = 0.02          # kg*m^2 (roll inertia)
Iyy = 0.02          # kg*m^2 (pitch inertia)
Izz = 0.04          # yaw inertia (not simulated here)

# Motor/ESC model
k_thrust = 1.6      # thrust coefficient [N per unit throttle]
motor_tau = 0.04    # motor time constant (s)

# ESC calibration errors (multiplicative gain per motor)
motor_gain = np.array([1.0, 0.95, 1.05, 0.98])  # motors 1..4

# Desired commands
z_target = 2.0      # meters
phi_target = 0.0    # roll (rad)
theta_target = 0.0  # pitch (rad)

# PID gains
alt_Kp, alt_Ki, alt_Kd = 25, 8, 10
att_Kp, att_Ki, att_Kd = 40, 2, 8

# ---------------- State Variables ----------------
z, z_dot = 0.0, 0.0
phi, phi_dot = 0.0, 0.0       # roll angle (rad)
theta, theta_dot = 0.0, 0.0   # pitch angle (rad)

# Integrators for PID
alt_int, phi_int, theta_int = 0.0, 0.0, 0.0
alt_prev_err, phi_prev_err, theta_prev_err = 0.0, 0.0, 0.0

# Motor throttle states (0..1)
motor_state = np.zeros(4)

# ---------------- Logging ----------------
z_hist = np.zeros(steps)
phi_hist = np.zeros(steps)
theta_hist = np.zeros(steps)
motor_hist = np.zeros((steps, 4))

# ---------------- Simulation Loop ----------------
for i in range(steps):
    # Altitude control (PID)
    z_err = z_target - z
    alt_int += z_err * dt
    alt_der = (z_err - alt_prev_err) / dt
    alt_prev_err = z_err
    collective = alt_Kp*z_err + alt_Ki*alt_int + alt_Kd*alt_der

    # Roll control (PID)
    phi_err = phi_target - phi
    phi_int += phi_err * dt
    phi_der = (phi_err - phi_prev_err) / dt
    phi_prev_err = phi_err
    roll_cmd = att_Kp*phi_err + att_Ki*phi_int + att_Kd*phi_der

    # Pitch control (PID)
    theta_err = theta_target - theta
    theta_int += theta_err * dt
    theta_der = (theta_err - theta_prev_err) / dt
    theta_prev_err = theta_err
    pitch_cmd = att_Kp*theta_err + att_Ki*theta_int + att_Kd*theta_der

    # Map control commands to motors
    # Motor layout: [FrontLeft, FrontRight, BackRight, BackLeft]
    u1 = collective - roll_cmd - pitch_cmd
    u2 = collective + roll_cmd - pitch_cmd
    u3 = collective + roll_cmd + pitch_cmd
    u4 = collective - roll_cmd + pitch_cmd
    cmd = np.array([u1, u2, u3, u4])

    # Normalize and apply ESC gain mismatch
    cmd = np.clip(cmd / 100.0, 0.0, 1.0)  # scale
    cmd = cmd * motor_gain

    # Motor dynamics (first-order response)
    motor_state += (cmd - motor_state) * (dt / motor_tau)

    # Thrust per motor
    thrusts = k_thrust * motor_state
    total_thrust = np.sum(thrusts)

    # Torques (roll from left-right imbalance, pitch from front-back imbalance)
    tau_phi = arm_length * ((thrusts[1] + thrusts[2]) - (thrusts[0] + thrusts[3]))
    tau_theta = arm_length * ((thrusts[2] + thrusts[3]) - (thrusts[0] + thrusts[1]))

    # Dynamics
    z_ddot = (total_thrust - mass*g) / mass
    phi_ddot = tau_phi / Ixx
    theta_ddot = tau_theta / Iyy

    z_dot += z_ddot * dt
    z += z_dot * dt
    phi_dot += phi_ddot * dt
    phi += phi_dot * dt
    theta_dot += theta_ddot * dt
    theta += theta_dot * dt

    # Prevent ground penetration
    if z < 0:
        z = 0
        z_dot = 0

    # Log
    z_hist[i] = z
    phi_hist[i] = np.degrees(phi)
    theta_hist[i] = np.degrees(theta)
    motor_hist[i, :] = motor_state

# ---------------- Metrics ----------------
def rise_time(time, signal, target):
    try:
        idx10 = np.argmax(signal >= 0.1 * target)
        idx90 = np.argmax(signal >= 0.9 * target)
        return time[idx90] - time[idx10]
    except:
        return None

rt = rise_time(t, z_hist, z_target)
overshoot = (np.max(z_hist) - z_target) / z_target * 100
ss_error = z_hist[-1] - z_target

print("Simulation Results:")
print(f" Target Altitude: {z_target:.2f} m")
print(f" Rise Time: {rt:.3f} s" if rt else "Rise Time: N/A")
print(f" Overshoot: {overshoot:.2f} %")
print(f" Steady-State Error: {ss_error:.3f} m")
print(f" Final Roll Angle: {phi_hist[-1]:.3f} deg")
print(f" Final Pitch Angle: {theta_hist[-1]:.3f} deg")

# ---------------- Plot Results ----------------
plt.figure(figsize=(10,8))

plt.subplot(3,1,1)
plt.plot(t, z_hist, label="Altitude (m)")
plt.axhline(z_target, color='r', linestyle='--', label="Target")
plt.ylabel("Altitude (m)")
plt.legend(); plt.grid(True)

plt.subplot(3,1,2)
plt.plot(t, phi_hist, label="Roll (deg)")
plt.plot(t, theta_hist, label="Pitch (deg)")
plt.ylabel("Angle (deg)")
plt.legend(); plt.grid(True)

plt.subplot(3,1,3)
for m in range(4):
    plt.plot(t, motor_hist[:,m], label=f"Motor {m+1}")
plt.ylabel("Throttle (0..1)")
plt.xlabel("Time (s)")
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()
