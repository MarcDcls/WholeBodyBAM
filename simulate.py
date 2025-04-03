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
MOTION = "squat motion"
# MOTION = "slow kick"
excluded_dofs = ["head_pitch", "head_yaw"]

compare_to_mujoco_postion = False

parser = argparse.ArgumentParser(description="Parser for simulate.py")
parser.add_argument("--log", "-l", help="Path to the log file", default=None)
parser.add_argument("--dir", "-d", help="Path to the directory containing the logs", default=None)
parser.add_argument("--kp", "-k", help="Proportional gain for the controllers", default=None, type=float)
parser.add_argument("--front", "-f", help="Lying on the floor face down at the start of the simulation", action="store_true")
parser.add_argument("--back", "-b", help="Lying on the floor face up at the start of the simulation", action="store_true")
parser.add_argument("--plot", "-p", help="Plot the selected dofs for each log", action="store_true")
parser.add_argument("--mae", "-m", help="Plot the MAE for each log", action="store_true")
parser.add_argument("--render", "-r", help="Render the simulation", action="store_true")
parser.add_argument("--vizualized", "-v", help="Vizualized model in the simulation", default="m4")
parser.add_argument("--simulate", "-s", help="Force to simulate the logs, even if simulated data already exist", action="store_true")
args = parser.parse_args()

if args.dir is None and (args.log is None or args.kp is None):
    raise ValueError("You must provide a log file or a directory")

if args.dir is None:
    logs = [[0, 1e10, args.log, args.kp]]
else:
    with open(args.dir + "/timings.json", "r") as f:
        timings = json.load(f)
    logs = [[value[0], value[1], args.dir + "/" + key, int(key[:2])] for key, value in timings.items()]
    
read_robot = placo.HumanoidRobot("model/urdf_low")
    
simulated_data = {}
for start, end, log, kp in logs:

    # Loading the simulated data if it exists
    simulated_path = f"{log[:-4]}_simulated.json"
    if os.path.exists(simulated_path) and not args.render and not args.simulate:
        with open(simulated_path, "r") as f:
            data = json.load(f)
        simulated_data[log[:-4]] = data

    else:
        print(f"Loading log {log}")

        history = placo.HistoryCollection()
        history.loadReplays(log)
        print(f"Log loaded, duration: from {history.smallestTimestamp()} to {history.biggestTimestamp()}")

        # Initialization
        target_robot = placo.HumanoidRobot("model/urdf_low")

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
        
        if compare_to_mujoco_postion:
            sim_mujoco = Simulator(use_bam=False, model_dir="position_model")

            for dof in read_robot.joint_names():
                sim_mujoco.set_control(dof, read_robot.get_joint(dof), True)

            sim_mujoco.set_T_world_site("left_foot", np.eye(4))
            sim_mujoco.t = 0

        sims = {"m1": {"sim": Simulator(use_bam=True, bam_model="m1", fancy_model=False, kp=kp, vin=15)}, 
                "m2": {"sim": Simulator(use_bam=True, bam_model="m2", fancy_model=False, kp=kp, vin=15)},
                "m3": {"sim": Simulator(use_bam=True, bam_model="m3", fancy_model=False, kp=kp, vin=15)},
                "m4": {"sim": Simulator(use_bam=True, bam_model="m4", fancy_model=False, kp=kp, vin=15)},
                "m5": {"sim": Simulator(use_bam=True, bam_model="m5", fancy_model=False, kp=kp, vin=15)},
                "m6": {"sim": Simulator(use_bam=True, bam_model="m6", fancy_model=False, kp=kp, vin=15)}}

        for key, sim in sims.items():            
            for dof in read_robot.joint_names():
                sim["sim"].set_control(dof, read_robot.get_joint(dof), True)

            if args.front:
                sim["sim"].set_T_world_site("trunk", T_world_trunk)
            elif args.back:
                sim["sim"].set_T_world_site("trunk", T_world_trunk)
            else:
                sim["sim"].set_T_world_site("left_foot", np.eye(4))

            sim["sim"].t = 0

        t_start = max(history.smallestTimestamp(), start)
        t_end = min(history.biggestTimestamp(), end)

        timesteps = []
        read_positions = {}
        goal_positions = {}
        sim_mujoco_positions = {}
        sims_positions = {key: {} for key in sims.keys()}
        for dof in read_robot.joint_names():
            read_positions[dof] = []
            goal_positions[dof] = []
            sim_mujoco_positions[dof] = []
            for key in sims.keys():
                if dof not in excluded_dofs:
                    sims_positions[key][dof] = []

        if args.render:
            sims[args.vizualized]["sim"].render(False)
            sims[args.vizualized]["sim"].viewer._cam.elevation = -15
            sims[args.vizualized]["sim"].viewer._cam.distance = 1.3
            sims[args.vizualized]["sim"].viewer._cam.lookat[2] = 0.2

        # Simulation loop
        t = history.smallestTimestamp()
        start_time = time.time()
        while t < history.biggestTimestamp():
            read_robot.read_from_histories(history, t, "read", False, np.zeros(26))
            target_robot.read_from_histories(history, t, "goal", False, np.zeros(26))

            if compare_to_mujoco_postion:
                for dof in read_robot.joint_names():
                    sim_mujoco.set_control(dof, target_robot.get_joint(dof))
                sim_mujoco.step()

            if t >= t_start and t < t_end:
                timesteps.append(t)
                for dof in read_robot.joint_names():
                    read_positions[dof].append(read_robot.get_joint(dof))
                    goal_positions[dof].append(target_robot.get_joint(dof))
                    if compare_to_mujoco_postion:
                        sim_mujoco_positions[dof].append(sim_mujoco.get_q(dof))

            for key, sim in sims.items():
                for dof in read_robot.joint_names():
                    sim["sim"].set_control(dof, target_robot.get_joint(dof))

                if key == args.vizualized and args.render:
                    sim["sim"].render(False)

                sim["sim"].step()

                if t >= t_start and t < t_end:
                    for dof in read_robot.joint_names():
                        if dof not in excluded_dofs:
                            sims_positions[key][dof].append(sim["sim"].get_q(dof))
                    
            t = sims[args.vizualized]["sim"].t + history.smallestTimestamp()
            if args.render:
                while time.time() - start_time < t - history.smallestTimestamp():
                    continue
        
        data = {"timesteps": timesteps,
                "read_positions": read_positions,
                "goal_positions": goal_positions,
                "sims_positions": sims_positions,
                "kp": kp}
        
        simulated_data[log[:-4]] = data

        # Saving the simulated data
        with open(f"{log[:-4]}_simulated.json", "w") as f:
            json.dump(data, f)

# Plotting dofs data
if args.plot:
    plot_dofs = ["left_ankle_roll", "left_ankle_pitch", "left_knee", "left_hip_roll", "left_hip_pitch"]
    for kp, data in simulated_data.items():
        for dof in plot_dofs:
            plt.plot(data["timesteps"], data["read_positions"][dof], label=f"read")
            plt.plot(data["timesteps"], data["goal_positions"][dof], label=f"goal")

            if compare_to_mujoco_postion:
                plt.plot(data["timesteps"], sim_mujoco_positions[dof], label=f"sim_mujoco")
        
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

    for log_name, data in simulated_data.items():
        KPs.append("Kp = " + str(data["kp"]))
        read_positions = data["read_positions"]
        
        for key, KEY in zip(["m1", "m2", "m3", "m4", "m5", "m6"], MAEs.keys()):
            sims_positions = data["sims_positions"][key]

            mae = 0
            for dof in read_robot.joint_names():
                if dof in excluded_dofs:
                    continue

                dof_mae = np.mean(np.abs(np.array(sims_positions[dof]) - np.array(read_positions[dof])))
                mae += dof_mae

                if dof not in MAEs_per_dof[KEY]:
                    MAEs_per_dof[KEY][dof] = []
                MAEs_per_dof[KEY][dof].append(dof_mae)
            
            mae /= (len(read_robot.joint_names()) - len(excluded_dofs))
            MAEs[KEY].append(mae)
    
    # Saving the MAEs
    data = {}
    for i, kp in enumerate(KPs):
        data[kp] = [{mae[0]: mae[1][i]} for mae in MAEs.items()]
    with open(dir_path + "maes.json", "w") as f:
        json.dump(data, f)

    # Plotting MAEs for each dof
    for dof in read_robot.joint_names():
        if dof in excluded_dofs:
            continue
        
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
        plt.title(f"MAE obtained while doing a {MOTION} for several Kps for {dof}")
        plt.savefig(dir_path + f"maes_{dof}.png")
        plt.close()

    # Plotting MAEs
    df = pandas.DataFrame(MAEs, index=KPs)
    ax = df.plot(kind="bar", figsize=(9, 4))
    ax.set_axisbelow(True)
    ax.grid(axis="y")
    ax.set_ylabel("MAE (rad)")
    plt.xticks(rotation=0, ha="center")
    plt.title(f"MAE obtained while doing a {MOTION} for several Kps")
    plt.savefig(dir_path + "maes_global.png")
    plt.show()
