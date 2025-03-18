import placo
import bam.mujoco as bMuj
import bam.model as bMod
from rhoban_rl.mujoco_simulator.simulator import Simulator
from placo_utils.visualization import robot_viz, frame_viz, point_viz, robot_frame_viz, footsteps_viz
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse

DT = 0.01

parser = argparse.ArgumentParser(description="Parser for read_log.py")
parser.add_argument("--log", "-l", help="Path to the log file", default="logs/squat_motion/12.log")
parser.add_argument("--kp", "-k", help="Proportional gain for the controllers", default=32, type=float)
parser.add_argument("--meshcat", "-m", help="Use meshcat for visualization", action="store_true")
parser.add_argument("--simulate", "-s", help="Simulate the robot in MuJoCo", action="store_true")
parser.add_argument("--render", "-r", help="Render the simulation", action="store_true")
parser.add_argument("--vizualized", "-v", help="Vizualized model in the simulation", default="m4")
args = parser.parse_args()

history = placo.HistoryCollection()
history.loadReplays(args.log)
print(f"Log loaded, duration: from {history.smallestTimestamp()} to {history.biggestTimestamp()}")

read_robot = placo.HumanoidRobot("model/sigmaban_2024")
target_robot = placo.HumanoidRobot("model/sigmaban_2024")

read_robot.read_from_histories(history, history.smallestTimestamp(), "read", False, np.zeros(26))
target_robot.read_from_histories(history, history.smallestTimestamp(), "goal", False, np.zeros(26))

if args.meshcat:
    viz = robot_viz(read_robot)
    viz.display(read_robot.state.q)
    time.sleep(3)

if args.simulate:
    sims = {"m1": {"sim": Simulator(model_dir="model")}, "m4": {"sim": Simulator(model_dir="model")}}

    mx106_actuators = ["left_ankle_roll", "right_ankle_roll",
                       "left_ankle_pitch", "right_ankle_pitch",
                       "left_knee", "right_knee",
                       "left_hip_roll", "right_hip_roll",
                       "left_hip_pitch", "right_hip_pitch",
                       "left_hip_yaw", "right_hip_yaw"]

    mx64_actuators = ["left_shoulder_pitch", "right_shoulder_pitch",
                      "left_shoulder_roll", "right_shoulder_roll",
                      "left_elbow", "right_elbow",
                      "head_pitch", "head_yaw"]
    
    for key, sim in sims.items():
        for dof in read_robot.joint_names():
            sim["sim"].set_q(dof, read_robot.get_joint(dof))

        sim["sim"].step()
        sim["sim"].set_T_world_site("left_foot", np.eye(4))
        sim["sim"].step()
        sim["sim"].t = 0

        mx106_model = bMod.load_model(f"../workspace/src/rhoban/bam/params/mx106/{key}.json")
        mx106_model.actuator.kp = args.kp
        sim["mx106_controller"] = bMuj.MujocoController(mx106_model, mx106_actuators, sim["sim"].model, sim["sim"].data)

        mx64_model = bMod.load_model(f"../workspace/src/rhoban/bam/params/mx64/{key}.json")
        mx64_model.actuator.kp = args.kp
        sim["mx64_controller"] = bMuj.MujocoController(mx64_model, mx64_actuators, sim["sim"].model, sim["sim"].data)

        for dof in read_robot.joint_names():
            sim[dof] = []

timesteps = []
read_positions = {}
goal_positions = {}
for dof in read_robot.joint_names():
    read_positions[dof] = []
    goal_positions[dof] = []

sims[args.vizualized]["sim"].render(True)
time.sleep(5)

t = history.smallestTimestamp()
start_time = time.time()
while t < history.biggestTimestamp():
    read_robot.read_from_histories(history, t, "read", False, np.zeros(26))
    target_robot.read_from_histories(history, t, "goal", False, np.zeros(26))

    timesteps.append(t)
    for dof in read_robot.joint_names():
        read_positions[dof].append(read_robot.get_joint(dof))
        goal_positions[dof].append(target_robot.get_joint(dof))

    if args.meshcat:
        viz.display(read_robot.state.q)
        time.sleep(DT)

    if args.simulate:
        for key, sim in sims.items():
            sim["mx106_controller"].update([target_robot.get_joint(dof) for dof in mx106_actuators])
            sim["mx64_controller"].update([target_robot.get_joint(dof) for dof in mx64_actuators])

            if key == args.vizualized and args.render:
                sim["sim"].render(True)

            sim["sim"].step()

            for dof in read_robot.joint_names():
                sim[dof].append(sim["sim"].get_q(dof))
    
    if args.simulate:
        t = sims["m1"]["sim"].t + history.smallestTimestamp()
    else:
        t += DT

    if args.simulate or args.meshcat:
        while time.time() - start_time < t - history.smallestTimestamp():
            continue
        
# Plotting
plot_dofs = ["right_knee"]
if args.simulate:

    for dof in plot_dofs:
        plt.plot(timesteps, read_positions[dof], label=f"read {dof}")
        plt.plot(timesteps, goal_positions[dof], label=f"goal {dof}")
    
        for key, sim in sims.items():
            plt.plot(timesteps, sim[dof], label=f"{key} {dof}")
    
        plt.title(f"{dof} position")
        plt.legend()
        plt.show()
        