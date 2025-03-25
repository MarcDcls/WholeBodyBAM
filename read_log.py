import placo
import bam.mujoco as bMuj
import bam.model as bMod
from rhoban_rl.mujoco_simulator.simulator import Simulator
from placo_utils.visualization import robot_viz, frame_viz, point_viz, robot_frame_viz, footsteps_viz
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse
import json
import pandas
import os

DT = 0.01

parser = argparse.ArgumentParser(description="Parser for read_log.py")
parser.add_argument("--log", "-l", help="Path to the log file", default=None)
parser.add_argument("--dir", "-d", help="Path to the directory containing the logs", default=None)
parser.add_argument("--kp", "-k", help="Proportional gain for the controllers", default=None, type=float)
parser.add_argument("--front", "-f", help="Lying on the floor face down at the start of the simulation", action="store_true")
parser.add_argument("--back", "-b", help="Lying on the floor face up at the start of the simulation", action="store_true")
parser.add_argument("--plot", "-p", help="Plot the selected dofs for each log", action="store_true")
parser.add_argument("--mae", "-m", help="Plot the MAE for each log", action="store_true")
parser.add_argument("--render", "-r", help="Render the simulation", action="store_true")
parser.add_argument("--vizualized", "-v", help="Vizualized model in the simulation", default="m4")
parser.add_argument("--title", "-t", help="Title of the plot", default="squat motion")
args = parser.parse_args()

if args.dir is None and (args.log is None or args.kp is None):
    raise ValueError("You must provide a log file or a directory")

if args.dir is None:
    logs = [[0, 1e10, args.log, args.kp]]
else:
    with open(args.dir + "/timings.json", "r") as f:
        timings = json.load(f)
    logs = [[value[0], value[1], args.dir + "/" + key, int(key[:-4])] for key, value in timings.items()]
    
read_robot = placo.HumanoidRobot("model/urdf")
    
simulated_data = {}
for start, end, log, kp in logs:

    # Loading the simulated data if it exists
    simulated_path = f"{log[:-4]}_simulated.json"
    if os.path.exists(simulated_path) and not args.render:
        with open(simulated_path, "r") as f:
            data = json.load(f)
        simulated_data[str(kp)] = data

    else:
        print(f"Loading log {log}")

        history = placo.HistoryCollection()
        history.loadReplays(log)
        print(f"Log loaded, duration: from {history.smallestTimestamp()} to {history.biggestTimestamp()}")

        # Initialization
        target_robot = placo.HumanoidRobot("model/urdf")

        read_robot.read_from_histories(history, history.smallestTimestamp(), "read", False, np.zeros(26))
        target_robot.read_from_histories(history, history.smallestTimestamp(), "goal", False, np.zeros(26))

        if args.front:
            T_world_trunk = np.array([[0, 0, 1, 0],
                                    [0, 1, 0, 0],
                                    [-1, 0, 0, 0.2],
                                    [0, 0, 0, 1]])
        elif args.back:
            T_world_trunk = np.array([[0, 0, -1, 0],
                                    [0, 1, 0, 0],
                                    [1, 0, 0, 0.2],
                                    [0, 0, 0, 1]])
            
        sims = {"m1": {"sim": Simulator(use_bam=True, model_dir="model/mujoco")}, 
                "m2": {"sim": Simulator(use_bam=True, model_dir="model/mujoco")},
                "m3": {"sim": Simulator(use_bam=True, model_dir="model/mujoco")},
                "m4": {"sim": Simulator(use_bam=True, model_dir="model/mujoco")},
                "m5": {"sim": Simulator(use_bam=True, model_dir="model/mujoco")},
                "m6": {"sim": Simulator(use_bam=True, model_dir="model/mujoco")}}

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

        for key, sim in sims.items():
            sim["sim"].add_actuator_model("mx106", f"../workspace/src/rhoban/bam/params/mx106/{key}.json", mx106_actuators, kp=kp)
            sim["sim"].add_actuator_model("mx64", f"../workspace/src/rhoban/bam/params/mx64/{key}.json", mx64_actuators, kp=kp)

            for dof in read_robot.joint_names():
                sim["sim"].set_control(dof, read_robot.get_joint(dof), True)

            sim["sim"].step()

            if args.front:
                sim["sim"].set_T_world_site("trunk", T_world_trunk)
                for i in range(1000):
                    sim["sim"].step()

            elif args.back:
                sim["sim"].set_T_world_site("trunk", T_world_trunk)
                for i in range(1000):
                    sim["sim"].step()
            
            else:
                sim["sim"].set_T_world_site("left_foot", np.eye(4))
                sim["sim"].step()

            sim["sim"].t = 0

        t_start = max(history.smallestTimestamp(), start)
        t_end = min(history.biggestTimestamp(), end)

        timesteps = []
        read_positions = {}
        goal_positions = {}
        sims_positions = {"m1": {}, "m2": {}, "m3": {}, "m4": {}, "m5": {}, "m6": {}}
        for dof in read_robot.joint_names():
            read_positions[dof] = []
            goal_positions[dof] = []
            for key in sims.keys():
                sims_positions[key][dof] = []

        if args.render:
            sims[args.vizualized]["sim"].render(True)

        # Simulation loop
        t = history.smallestTimestamp()
        start_time = time.time()
        while t < history.biggestTimestamp():
            read_robot.read_from_histories(history, t, "read", False, np.zeros(26))
            target_robot.read_from_histories(history, t, "goal", False, np.zeros(26))

            if t >= t_start and t < t_end:
                timesteps.append(t)
                for dof in read_robot.joint_names():
                    read_positions[dof].append(read_robot.get_joint(dof))
                    goal_positions[dof].append(target_robot.get_joint(dof))

            for key, sim in sims.items():
                for dof in read_robot.joint_names():
                    sim["sim"].set_control(dof, target_robot.get_joint(dof))

                if key == args.vizualized and args.render:
                    sim["sim"].render(True)

                sim["sim"].step()

                if t >= t_start and t < t_end:
                    for dof in read_robot.joint_names():
                        sims_positions[key][dof].append(sim["sim"].get_q(dof))
                    
            t = sims["m1"]["sim"].t + history.smallestTimestamp()
            if args.render:
                while time.time() - start_time < t - history.smallestTimestamp():
                    continue
        
        data = {"timesteps": timesteps,
                "read_positions": read_positions,
                "goal_positions": goal_positions,
                "sims_positions": sims_positions}
        
        simulated_data[str(kp)] = data

        # Saving the simulated data
        with open(f"{log[:-4]}_simulated.json", "w") as f:
            json.dump(data, f)

# Plotting dofs data
if args.plot:
    plot_dofs = ["right_ankle_roll"]

    for kp, data in simulated_data.items():
        for dof in plot_dofs:
            plt.plot(data["timesteps"], data["read_positions"][dof], label=f"read")
            plt.plot(data["timesteps"], data["goal_positions"][dof], label=f"goal")
        
            for key, positions in data["sims_positions"].items():
                plt.plot(data["timesteps"], positions[dof], label=f"{key}")

            plt.title(f"{dof} position for {log}")
            plt.legend()
            plt.show()

# Calculating MAEs
if args.mae:
    dir_path = ""
    if args.dir is not None:
        dir_path = args.dir + "/mae/"

    MAEs = {"M1": [], "M2": [], "M3": [], "M4": [], "M5": [], "M6": []}
    MAEs_per_dof = {"M1": {}, "M2": {}, "M3": {}, "M4": {}, "M5": {}, "M6": {}}
    KPs = []

    for kp, data in simulated_data.items():
        KPs.append("Kp = " + kp)
        read_positions = data["read_positions"]
        
        for key, KEY in zip(["m1", "m2", "m3", "m4", "m5", "m6"], MAEs.keys()):
            sims_positions = data["sims_positions"][key]

            mae = 0
            for dof in read_robot.joint_names():
                dof_mae = np.mean(np.abs(np.array(sims_positions[dof]) - np.array(read_positions[dof])))
                mae += dof_mae

                if dof not in MAEs_per_dof[KEY]:
                    MAEs_per_dof[KEY][dof] = []
                MAEs_per_dof[KEY][dof].append(dof_mae)
            
            mae /= len(read_robot.joint_names())
            MAEs[KEY].append(mae)
    
    # Saving the MAEs
    data = {}
    for i, kp in enumerate(KPs):
        data[kp] = [{mae[0]: mae[1][i]} for mae in MAEs.items()]
    with open(dir_path + "maes.json", "w") as f:
        json.dump(data, f)

    # Plotting MAEs for each dof
    for dof in read_robot.joint_names():
        
        plot_data = {"M1": [], "M2": [], "M3": [], "M4": [], "M5": [], "M6": []}
        for key in plot_data.keys():
            for i, kp in enumerate(KPs):
                plot_data[key].append(MAEs_per_dof[key][dof][i])

        df = pandas.DataFrame(plot_data, index=KPs)
        ax = df.plot(kind="bar", figsize=(9, 4))
        ax.set_axisbelow(True)
        ax.grid(axis="y")
        ax.set_ylabel("MAE (rad)")
        plt.xticks(rotation=0, ha="center")
        plt.title(f"MAE obtained while doing a {args.title} for several KPs for {dof}")
        plt.savefig(dir_path + f"maes_{dof}.png")
        plt.close()

    # Plotting MAEs
    df = pandas.DataFrame(MAEs, index=KPs)
    ax = df.plot(kind="bar", figsize=(9, 4))
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    ax.set_ylabel("MAE (rad)")
    plt.xticks(rotation=0, ha="center")
    plt.title(f"MAE obtained while doing a {args.title} for several KPs")
    plt.savefig(dir_path + "maes_global.png")
    plt.show()
