import placo
from rhoban_rl.mujoco_simulator.simulator import Simulator
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import json
import pandas
import os

DT = 0.01

parser = argparse.ArgumentParser(description="Parser for drop.py")
parser.add_argument("--log", "-l", help="Path to the log file", default="logs/drop_kp/32_to_16.log")
parser.add_argument("--render", "-r", help="Render the simulation", action="store_true")
args = parser.parse_args()

read_robot = placo.HumanoidRobot("../workspace/sigmaban/sigmaban")
target_robot = placo.HumanoidRobot("../workspace/sigmaban/sigmaban")

history = placo.HistoryCollection()
history.loadReplays(args.log)
print(f"Log loaded, duration: from {history.smallestTimestamp()} to {history.biggestTimestamp()}")

read_robot.read_from_histories(history, history.smallestTimestamp(), "read", False, np.zeros(26))
target_robot.read_from_histories(history, history.smallestTimestamp(), "goal", False, np.zeros(26))

mx106_actuators = ["left_ankle_roll", "right_ankle_roll",
                "left_ankle_pitch", "right_ankle_pitch",
                "left_knee", "right_knee",
                "left_hip_roll", "right_hip_roll",
                "left_hip_pitch", "right_hip_pitch"]

mx64_actuators = ["left_shoulder_pitch", "right_shoulder_pitch",
                "left_shoulder_roll", "right_shoulder_roll",
                "left_elbow", "right_elbow",
                "head_pitch", "head_yaw",
                "left_hip_yaw", "right_hip_yaw"]

data = {"read": 5}
for model in ["m1", "m2", "m3", "m4", "m5", "m6"]:
    print(f"Testing model {model}")

    for kp in range(1, 12)[::-1]:
        print(f"Kp = {kp}")

        sim = Simulator(use_bam=True, bam_model=model, kp=32, vin=15)

        # Setting the initial position
        for dof in read_robot.joint_names():
            sim.set_control(dof, read_robot.get_joint(dof), True)
        sim.set_T_world_site("left_foot", np.eye(4))
        sim.t = 0

        t0 = time.time()
        fallen = False
        drop = False
        while sim.t < 12:
            if args.render:
                sim.render(False)

            sim.step()

            pressure = sim.get_pressure_sensors()
            pressure_sum = np.sum(pressure["left"]) + np.sum(pressure["right"])
            if pressure_sum < 20:
                fallen = True
                break

            # Dropping the Kp
            if not drop and sim.t > 3:
                sim.set_kp(kp)

            if args.render:
                while time.time() - t0 < sim.t:
                    continue
        
        if args.render:
            sim.viewer.close()
        
        if fallen:
            data[model] = kp
            break

print(data)
        
