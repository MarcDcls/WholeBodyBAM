import placo
from rhoban_rl.mujoco_simulator.simulator import Simulator
from placo_utils.visualization import robot_viz
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse

DT = 0.01

parser = argparse.ArgumentParser(description="Parser for read_log.py")
parser.add_argument("--log", "-l", help="Path to the log file", default=None)
parser.add_argument("--kp", "-k", help="Proportional gain for the controllers", default=None, type=float)
parser.add_argument("--plot", "-p", help="Plot the selected dofs for each log", action="store_true")
parser.add_argument("--meshcat", "-m", help="Use meshcat for visualization", action="store_true")
parser.add_argument("--simulate", "-s", help="Simulate the robot in MuJoCo", action="store_true")
parser.add_argument("--render", "-r", help="Render the simulation", action="store_true")
parser.add_argument("--vizualized", "-v", help="Vizualized model in the simulation", default="m4")
args = parser.parse_args()

history = placo.HistoryCollection()
history.loadReplays(args.log)
print(f"Log loaded, duration: from {history.smallestTimestamp()} to {history.biggestTimestamp()}")

read_robot = placo.HumanoidRobot("model/urdf")
target_robot = placo.HumanoidRobot("model/urdf")

# Initialization
read_robot.read_from_histories(history, history.smallestTimestamp(), "read", False, np.zeros(26))
target_robot.read_from_histories(history, history.smallestTimestamp(), "goal", False, np.zeros(26))

if args.meshcat:
    viz = robot_viz(read_robot)
    viz.display(read_robot.state.q)
    time.sleep(3)

if args.simulate:
    sim = Simulator(use_bam=True, model_dir="model/mujoco", config="model/bam_config.json", kp=args.kp)
    for dof in read_robot.joint_names():
        sim.set_control(dof, read_robot.get_joint(dof), True)

    sim.step()
    sim.set_T_world_site("left_foot", np.eye(4))
    sim.step()
    sim.t = 0

timesteps = []
read_positions = {}
goal_positions = {}
sim_positions = {}
for dof in read_robot.joint_names():
    read_positions[dof] = []
    goal_positions[dof] = []
    sim_positions[dof] = []

if args.render:
    sim.render(True)

# Simulation loop
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
        for dof in read_robot.joint_names():
            sim.set_control(dof, target_robot.get_joint(dof))

        if args.render:
            sim.render(True)

        sim.step()

        for dof in read_robot.joint_names():
            sim_positions[dof].append(sim.get_q(dof))

    if args.simulate:
        t = sim.t + history.smallestTimestamp()
    else:
        t += DT

    if args.render or args.meshcat:
        while time.time() - start_time < t - history.smallestTimestamp():
            continue

# Plotting
plot_dofs = ["right_knee"]
if args.simulate:
    if args.plot:
        for dof in plot_dofs:
            plt.plot(timesteps, read_positions[dof], label=f"read {dof}")
            plt.plot(timesteps, goal_positions[dof], label=f"goal {dof}")
            plt.plot(timesteps, sim_positions[dof], label=f"sim {dof}")
            plt.title(f"{dof} position")
            plt.legend()
            plt.show()
