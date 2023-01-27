import os
import pandas as pd
from glob import glob
from typing import List, Dict, Tuple, cast
from typing_extensions import Literal
import numpy as np
from scipy import signal
from pyomo.environ import value as pyovalue
import acinoset_misc as misc
import acinoset_opt as opt
import shared.physical_education as pe
from py_utils import data_ops
from acinoset_models import PoseModelGMM
import json


def check_grf(robot: pe.system.System3D):
    # This confirms that IPOPT is configured to ensure that one of the following is 0 - |x| = x^{+} + x^{-}.
    for foot in pe.foot.feet(robot):
        grfxy = foot["GRFxy"]
        for fe, cp in robot.indices(one_based=True):
            if grfxy[fe, 1, 0].value == 0.0 or grfxy[fe, 1, 1].value == 0.0:
                # NOP - correct
                pass
            else:
                print(f"Invalid state: {foot.name} ({fe})")
            if grfxy[fe, 1, 2].value == 0.0 or grfxy[fe, 1, 3].value == 0.0:
                # NOP - correct
                pass
            else:
                print(f"Invalid state 2: {foot.name} ({fe})")


def plot_cost_functions():
    import matplotlib.pyplot as plt
    redesc_a, redesc_b, redesc_c = 3, 10, 20
    #Plot
    r_x = np.arange(-30, 30, 1e-1)
    r_y1 = [misc.redescending_loss(i, redesc_a, redesc_b, redesc_c) for i in r_x]
    r_y2 = abs(r_x)
    r_y3 = r_x**2
    fig = plt.figure(figsize=(6, 4), dpi=120)
    plt.plot(r_x, r_y3, label='Quadratic')
    plt.plot(r_x, r_y1, label='Redescending')
    plt.xlabel("Error (pixels)")
    plt.ylabel("Cost Value")
    ax = plt.gca()
    ax.set_ylim((-5, 50))
    ax.legend()
    plt.show(block=False)
    fig.savefig("./data/cost-function.pdf", bbox_inches="tight")
    plt.close()


def example_robustness(root_dir: str, dir_prefix: str):
    import matplotlib.pyplot as plt
    metrics = {
        "single_traj_error": [],
        "data_driven_traj_error": [],
        "physics_based_traj_error": [],
    }
    test_run = ("phantom", "2019_03_07", "run")
    cam_space = ((0, 1, 2, 3, 4, 5), )
    for cam_idx in cam_space[0]:
        # Configure trajectory.
        cheetah = test_run[0]
        date = test_run[1]
        trial = test_run[2]
        data_path = f"{date}/{cheetah}/{trial}"
        print(f"{data_path}:")
        # Read the scene data file and the reconstruction data.
        multi_view_data = data_ops.load_pickle(os.path.join(dir_prefix, data_path, "fte_kinematic", "fte.pickle"))
        # Ensure that the camera has been collected.
        if not (os.path.exists(os.path.join(dir_prefix, data_path, f"fte_kinematic_{cam_idx}"))
                and os.path.exists(os.path.join(dir_prefix, data_path, f"fte_kinematic_orig_{cam_idx}"))):
            continue
        single_view_data = data_ops.load_pickle(
            os.path.join(dir_prefix, data_path, f"fte_kinematic_orig_{cam_idx}", "fte.pickle"))
        data_driven_data = data_ops.load_pickle(
            os.path.join(dir_prefix, data_path, f"fte_kinematic_{cam_idx}", "fte.pickle"))
        physics_based_traj_error = []
        for i in range(1):
            physics_based_data = data_ops.load_pickle(
                os.path.join(dir_prefix, data_path, f"fte_kinetic_{cam_idx}",
                             f"fte{i}.pickle" if i > 0 else "fte.pickle"))
            _, physics_based_traj_error_tmp, _ = misc.traj_error(multi_view_data["positions"].copy(),
                                                                 physics_based_data["positions"].copy(),
                                                                 "physics-based model",
                                                                 centered=True)
            physics_based_traj_error.append(physics_based_traj_error_tmp)

        physics_based_traj_error = np.mean(physics_based_traj_error, axis=0)
        # physics_based_traj_error_std = np.std(physics_based_traj_error, axis=0)
        # Calculate the trajectory error per bodypart.
        _, single_traj_error, _ = misc.traj_error(multi_view_data["positions"].copy(),
                                                  single_view_data["positions"].copy(),
                                                  centered=True)
        _, data_driven_traj_error, _ = misc.traj_error(multi_view_data["positions"].copy(),
                                                       data_driven_data["positions"].copy(),
                                                       "data-driven model",
                                                       centered=True)
        metrics["single_traj_error"].append(float(single_traj_error.mean()))
        metrics["data_driven_traj_error"].append(float(data_driven_traj_error.mean()))
        metrics["physics_based_traj_error"].append(float(physics_based_traj_error.mean()))

    fig = plt.figure(figsize=(16, 12), dpi=120)
    scenarios = ("1", "2", "3", "4", "5", "6")
    width = 0.25  # the width of the bars
    x = np.arange(len(scenarios))
    rect1 = plt.bar(x, metrics["single_traj_error"], width, label="Default", color=misc.plot_color["charcoal"])
    rect2 = plt.bar(x + width,
                    metrics["data_driven_traj_error"],
                    width,
                    label="Data-driven",
                    color=misc.plot_color["green"])
    rect3 = plt.bar(x + 2 * width,
                    metrics["physics_based_traj_error"],
                    width,
                    label="Physics-based",
                    color=misc.plot_color["orange"])
    plt.xticks(x + width, scenarios)
    plt.ylabel("MPJPE (mm)")
    plt.xlabel("Camera")
    plt.legend()
    fig.savefig(os.path.join(dir_prefix, "example-cam-robustness.pdf"), bbox_inches="tight")
    plt.close()


def check_joint_estimation(root_dir: str, dir_prefix: str):
    cheetah = "shiraz"
    date = "2009_09_08"
    trial = "04"
    data_path = f"kinetic_dataset/{date}/{cheetah}/trial{trial}"
    fte_gt = data_ops.load_pickle(os.path.join(dir_prefix, data_path, "fte_kinetic", "fte.pickle"))
    fte = data_ops.load_pickle(os.path.join(root_dir, data_path, "fte_kinetic", "fte.pickle"))
    single_view_mpjpe_mm, single_view_error, _ = misc.traj_error(fte_gt["positions"].copy(), fte["positions"].copy())
    tau_gt = [list(k.ravel()) for k in fte_gt['tau'].values()]
    tau = [list(k.ravel()) for k in fte['tau'].values()]
    print(f"Torque RMSE: {misc.rmse(np.array(pe.utils.flatten(tau_gt)), np.array(pe.utils.flatten(tau)))}")
    import matplotlib.pyplot as plt
    from cycler import cycler
    titles = ("Tail base", "Left back hip", "Right back hip", "Spine", "Left front hip", "Right front hip", "Neck",
              "Tail mid", "Left front knee", "Left front ankle", "Right front knee", "Right front ankle",
              "Left back knee", "Left back ankle", "Right back knee", "Right back ankle")
    fig = plt.figure(figsize=(12, 32), dpi=120)
    axs = fig.subplots(8, 2)
    axs = axs.flatten()
    for i, motor in enumerate(fte_gt['tau'].keys()):
        axs[i].set_prop_cycle(
            cycler('color', [misc.plot_color["orange"], misc.plot_color["green"], misc.plot_color["red"]]))
        axs[i].plot(fte_gt['tau'][motor], alpha=0.3)
        axs[i].set_prop_cycle(
            cycler('color', [misc.plot_color["orange"], misc.plot_color["green"], misc.plot_color["red"]]))
        axs[i].plot(fte['tau'][motor])
        tau_err = misc.rmse(np.array(pe.utils.flatten(fte_gt['tau'][motor])),
                            np.array(pe.utils.flatten(fte['tau'][motor])))
        axs[i].set_title(f"{titles[i]} ({tau_err:.3f})")
    # fig.savefig(os.path.join(out_dir, "torque-profile.pdf"), bbox_inches="tight")
    plt.show(block=False)
    plt.close()


def contact_detection_analysis(root_dir: str, dir_prefix: str, kinetic_dataset: bool = False, monocular: bool = False):
    # Need to get the 3rd camera for this set: ("arabia", "2009_09_08", "15")
    # Can't get reliable kinematics for ("shiraz", "2009_09_08", "02") Not a major issue becaues the GRF data is corrupted for this test.
    if kinetic_dataset:
        test_set = (("arabia", "2009_09_07", "06"), ("shiraz", "2009_09_07", "04"), ("shiraz", "2009_09_08", "04"),
                    ("shiraz", "2009_09_11", "01"), ("shiraz", "2009_09_11", "02"))
    else:
        test_set = (
            ("jules", "2017_12_09/bottom", "flick2"),
            ("jules", "2019_03_09", "flick1"),
            ("phantom", "2019_03_03", "run"),  # curve run (acceleration)
            ("phantom", "2017_08_29/top", "run1_1"),  # straight run (steady-state slow)
            # ("phantom", "2017_12_09/bottom", "run2"),
            ("jules", "2017_08_29/top", "run1_1"),  # straight run (steady-state fast)
            ("jules", "2017_09_02/top", "run1"),  # straight run (steady-state fast)
            ("phantom", "2017_09_02/top", "run1_2"),  # straight run (deceleration)
            ("phantom", "2019_03_07", "run"),  # straight run (steady-state slow)
            ("jules", "2017_08_29/top", "run1_2"),  # straight run (deceleration/trot)
            ("jules", "2017_09_02/bottom", "run2"))  # straight run (steady-state medium)
    num_false_positives = 0
    num_missed = 0
    num_false_positives_tmp = 0
    num_gt_contacts = 0
    num_est_contacts = 0
    num_est_contacts_tmp = 0
    num_error = 0
    num_error_tmp = 0
    num_matched = 0
    num_matched_tmp = 0
    results = {
        "total_contacts_points": 0,
        "total_estimated_contacts_points": [0, 0],
        "total_false_positives": [0, 0],
        "total_missed": 0,
    }
    for test_run in test_set:
        # Configure trajectory.
        cheetah = test_run[0]
        date = test_run[1]
        trial = test_run[2]
        data_path = f"kinetic_dataset/{date}/{cheetah}/trial{trial}" if kinetic_dataset else f"{date}/{cheetah}/{trial}"
        estimator = opt.init_trajectory(root_dir,
                                        data_path,
                                        cheetah,
                                        include_camera_constraints=False,
                                        kinetic_dataset=kinetic_dataset,
                                        monocular_enable=monocular,
                                        kinematic_model=False)
        print(f"Processing {data_path}")
        robot = estimator.model
        # Init from previous run FTE.
        data_dir = os.path.join(dir_prefix, estimator.params.data_dir.split("/cheetah_videos/")[1])
        fte_states = data_ops.load_pickle(
            os.path.join(
                data_dir, "fte_kinematic" if estimator.scene.cam_idx is None or
                (not monocular) else f"fte_kinematic_{estimator.scene.cam_idx}", "fte.pickle"))
        init_q = fte_states["q"][:len(robot.m.fe), :]
        init_dq = fte_states["dq"][:len(robot.m.fe), :]
        init_ddq = fte_states["ddq"][:len(robot.m.fe), :]
        estimator.com_vel = fte_states["com_vel"]
        estimator.com_pos = fte_states["com_pos"]
        for fe, cp in robot.indices(one_based=True):
            p = 0
            for link in robot.links:
                for q in link.pyomo_sets["q_set"]:
                    robot[link.name]["q"][fe, cp, q].value = init_q[fe - 1, p]
                    robot[link.name]["dq"][fe, cp, q].value = init_dq[fe - 1, p]
                    robot[link.name]["ddq"][fe, cp, q].value = init_ddq[fe - 1, p]
                    p += 1
        speed = np.linalg.norm(estimator.com_vel, axis=1)  # type: ignore
        cheetah_speed_mps = cast(float, np.mean(speed))
        contacts, contacts_tmp = misc.contact_detection(estimator.model,
                                                        estimator.params.start_frame,
                                                        cheetah_speed_mps,
                                                        estimator.scene.fps,
                                                        data_dir,
                                                        plot=False)
        with open(os.path.join(estimator.params.data_dir, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
        gt_contacts = metadata["contacts"]
        results[data_path] = {"speed": cheetah_speed_mps}
        for foot in pe.foot.feet(estimator.model):
            gt_foot_contact = gt_contacts[foot.name]
            foot_contact = contacts[foot.name]
            foot_contact_tmp = contacts_tmp[foot.name]
            if gt_foot_contact is not None:
                gt_stance = list(range(gt_foot_contact[0][0], gt_foot_contact[0][1]))
                num_gt_contacts += len(gt_stance)
                # Determine whether there is a detection.
                if foot_contact is not None:
                    results[data_path][foot.name] = {
                        "stance_length": foot_contact[0][1] - foot_contact[0][0],
                        "limb": foot_contact[0][3],
                        "missed": 0,
                        "matched": 0,
                        "len_diff": 0,
                        "start_diff": 0,
                        "end_diff": 0
                    }
                    estimated_stance = list(range(foot_contact[0][0], foot_contact[0][1]))
                    num_est_contacts += len(estimated_stance)
                    combined = len(np.unique(estimated_stance + gt_stance))
                    matched = len(np.intersect1d(estimated_stance, gt_stance))
                    # error = np.abs(combined - matched)
                    error = np.abs(len(estimated_stance) - matched)
                    num_error += error
                    num_matched += matched
                    results[data_path][foot.name]["missed"] = error
                    results[data_path][foot.name]["matched"] = matched
                    results[data_path][foot.name]["len_diff"] = len(gt_stance) - len(estimated_stance)
                    results[data_path][foot.name]["start_diff"] = gt_foot_contact[0][0] - foot_contact[0][0]
                    results[data_path][foot.name]["end_diff"] = gt_foot_contact[0][1] - foot_contact[0][1]
                else:
                    print(f"Missed of contact for {data_path}: {foot.name}")
                    num_missed += 1
                if foot_contact_tmp is not None:
                    estimated_stance = list(range(foot_contact_tmp[0][0], foot_contact_tmp[0][1]))
                    num_est_contacts_tmp += len(estimated_stance)
                    combined = len(np.unique((estimated_stance, gt_stance)))
                    matched = len(np.intersect1d(estimated_stance, gt_stance))
                    # num_error_tmp += np.abs(combined - matched)
                    num_error_tmp += np.abs(len(estimated_stance) - matched)
                    num_matched_tmp += matched
            else:
                # Check whether there is not a false positive.
                if foot_contact is not None:
                    num_false_positives += 1
                if foot_contact_tmp is not None:
                    num_false_positives_tmp += 1
        results["total_contacts_points"] = num_gt_contacts
        results["total_missed"] = num_missed
        results["total_estimated_contacts_points"] = [num_est_contacts, num_est_contacts_tmp]
        results["total_false_positives"] = [num_false_positives, num_false_positives_tmp]
        results["total_contact_missed"] = [num_error, num_error_tmp]
        results["total_contact_matched"] = [num_matched, num_matched_tmp]
    data_ops.save_pickle(
        os.path.join(dir_prefix, "contact_detection_monocular.pickle" if monocular else "contact_detection.pickle"),
        results)
    disp_results = {
        "contact_points": results["total_contacts_points"],
        "est_contact_points": results["total_estimated_contacts_points"][0],
        "est_contact_points_2": results["total_estimated_contacts_points"][1],
        "missed_detection": results["total_missed"],
        "missed_contact": results["total_contact_missed"][0],
        "missed_contact_2": results["total_contact_missed"][1],
        "matched_contact": results["total_contact_matched"][0],
        "matched_contact_2": results["total_contact_matched"][1],
        "false_positives": results["total_false_positives"][0],
        "false_positives_2": results["total_false_positives"][1]
    }
    df = pd.DataFrame(disp_results, index=[0])
    # Kinetic dataset uncertainty in result 5 runs * +-1 for each touchdown and takeoff events (1 stride for each run). Therefore, 5*1*4 = 10%.
    # AcinoSet dataset uncertainty in result 10 runs * +-1 for each touchdown and takeoff events (1 stride for each run). Therefore, 10*1*4 = 13.6%
    print(df)
    print(results)


def animate_torque_plot(robot: pe.system.System3D):
    import matplotlib.pyplot as plt
    from matplotlib import animation
    total_mass = sum(link.mass for link in robot.links)
    scale_forces_by = total_mass * 9.81
    torque = {}
    for motor in pe.motor.torques(robot):
        torque[motor.name] = [
            scale_forces_by * tau.value
            for tau in [motor.pyomo_vars["Tc"][fe, idx] for fe in robot.m.fe for idx in motor.pyomo_sets["Tc_set"]]
        ]
    x1 = torque["front-left-hip-pitch"]
    x2 = torque["LFL_HFL_torque"]
    nfe = len(x1)
    total_time = pe.utils.total_time(robot.m)
    time_steps = np.linspace(0, total_time, num=nfe)
    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig_idx = 1
    ax = plt.subplot(1, 1, fig_idx)
    ax.plot(time_steps, x1, label="Front Left Shoulder", color=misc.plot_color["red"])
    ax.plot(time_steps, x2, label="Front Left Carpus", color=misc.plot_color["blue"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel('Torque (Nm)')
    ax.legend()
    fig_idx += 1
    plt.show(block=False)

    def animate(i):
        ax.cla()  # clear the previous image
        ax.plot(time_steps[:i], x1[:i])  # plot the line
        ax.plot(time_steps[:i], x2[:i])  # plot the line
        ax.set_xlim([0, time_steps[-1]])  # fix the x axis
        ax.set_ylim([1.1 * np.min(x1), 1.1 * np.max(x1)])  # fix the y axis

    anim = animation.FuncAnimation(fig, animate, frames=len(x) + 1, interval=1, blit=False)
    plt.show(block=False)


def get_power_values(robot: pe.system.System3D, force_scale: float) -> List[np.ndarray]:
    nfe = len(robot.m.fe)
    power_arr = []

    for motor in pe.motor.torques(robot):
        _power = np.array([pyovalue(P) * force_scale for P in pe.motor.power(motor, robot.pyo_variables)]).reshape(
            (nfe, -1))

        power_arr.append(_power)

    return power_arr


def determine_dlc_performance():
    # Arabia labeled run.
    points_2d_df = misc.load_dlc_points_as_df([
        "/data/zico/cheetah_videos/kinetic_dataset/2009_09_07/arabia/trial06/dlc/cam1DLC_resnet152_CheetahOct14shuffle3_600000.h5",
        "/data/zico/cheetah_videos/kinetic_dataset/2009_09_07/arabia/trial06/dlc/cam2DLC_resnet152_CheetahOct14shuffle3_600000.h5"
    ],
                                              verbose=False)
    points_2d_df = points_2d_df[points_2d_df["frame"] < 199]
    points_2d_df.loc[points_2d_df['likelihood'] <= 0.5, ["x", "y"]] = np.nan  # ignore points with low likelihood
    gt_points_2d_df = misc.load_dlc_points_as_df([
        "/data/zico/cheetah_videos/kinetic_dataset/2009_09_07/arabia/trial06/dlc_hand_labeled/cam1.h5",
        "/data/zico/cheetah_videos/kinetic_dataset/2009_09_07/arabia/trial06/dlc_hand_labeled/cam2.h5"
    ],
                                                 hand_labeled=True,
                                                 verbose=False)
    y_actual1 = gt_points_2d_df[["x", "y"]].to_numpy(dtype=np.float32)
    y_predicted1 = points_2d_df[["x", "y"]].to_numpy(dtype=np.float32)

    # Shiraz labeled run.
    points_2d_df = misc.load_dlc_points_as_df([
        "/data/zico/cheetah_videos/kinetic_dataset/2009_09_07/shiraz/trial04/dlc/cam1DLC_resnet152_CheetahOct14shuffle3_600000.h5",
        "/data/zico/cheetah_videos/kinetic_dataset/2009_09_07/shiraz/trial04/dlc/cam2DLC_resnet152_CheetahOct14shuffle3_600000.h5"
    ],
                                              verbose=False)
    points_2d_df = points_2d_df[points_2d_df["frame"] < 199]
    points_2d_df.loc[points_2d_df['likelihood'] <= 0.5, ["x", "y"]] = np.nan  # ignore points with low likelihood
    gt_points_2d_df = misc.load_dlc_points_as_df([
        "/data/zico/cheetah_videos/kinetic_dataset/2009_09_07/shiraz/trial04/dlc_hand_labeled/cam1.h5",
        "/data/zico/cheetah_videos/kinetic_dataset/2009_09_07/shiraz/trial04/dlc_hand_labeled/cam2.h5"
    ],
                                                 hand_labeled=True,
                                                 verbose=False)
    y_actual2 = gt_points_2d_df[["x", "y"]].to_numpy(dtype=np.float32)
    y_predicted2 = points_2d_df[["x", "y"]].to_numpy(dtype=np.float32)

    y_actual = np.vstack((y_actual1, y_actual2))
    y_predicted = np.vstack((y_predicted1, y_predicted2))

    print(f"RMSE (pixels): {rmse(y_actual, y_predicted):.2f}")
    print(f"MAD: {mad(y_actual, y_predicted):.2f}")
    print(f"Mean: {np.nanmean(y_actual - y_predicted):.2f}")
    print(f"STD: {std_dev(y_actual, y_predicted):.2f}")
    print(f"# predictions: {y_predicted[~np.isnan(y_predicted)].shape[0]}")
    print(f"# GT labels: {y_actual[~np.isnan(y_actual)].shape[0]}")

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 9), dpi=120)
    plt.hist((y_actual - y_predicted).flatten(), density=True, bins=100, color="b")  # you may select the no. of bins
    plt.xlabel('Residuals', fontsize=15)
    plt.ylabel('Density', fontsize=15)
    plt.show()


def rmse(predictions, targets):
    differences = predictions - targets
    differences_squared = differences**2
    mean_of_differences_squared = np.nanmean(differences_squared.flatten())
    rmse_val = np.sqrt(mean_of_differences_squared)
    return rmse_val


def mad(predictions, targets):
    differences = predictions - targets
    median = np.nanmedian(differences.flatten())
    mad_val = np.nanmedian(np.abs(differences - median))
    return mad_val


def std_dev(predictions, targets):
    differences = predictions - targets
    std_val = np.nanstd(differences.flatten())
    return std_val


"""
shiraz_0804_contact = {
    # "forelimb-trailing": ["right", 32, 40],
    "forelimb-trailing": ["right", 0, 0],
    "forelimb-leading": ["left", 0, 0 ],
    "hindlimb-leading": ["left", 10, 25],
    "hindlimb-trailing": ["right", 2, 18]
}
"""


def contact_json_conversion(robot: pe.system.System3D, json_path: str):
    with open(json_path, "r") as f:
        contact_json = json.load(f)
    start_frame = contact_json["start_frame"]
    end_frame = contact_json["end_frame"]
    foot_contact_order = contact_json["contacts"]
    ret = {
        "forelimb-trailing": ["", 0, 0],
        "forelimb-leading": ["", 0, 0],
        "hindlimb-leading": ["", 0, 0],
        "hindlimb-trailing": ["", 0, 0]
    }
    for foot in pe.foot.feet(robot):
        limb = "forelimb" if foot.name[1] == "F" else "hindlimb"
        side = "right" if foot.name[2] == "R" else "left"
        if foot.name in foot_contact_order and foot_contact_order[foot.name] is not None:
            data = foot_contact_order[foot.name]
            start_idx = data[0][0] - start_frame
            end_idx = data[0][1] - start_frame
            if data[0][1] > end_frame:
                # Not the full stance period, so remove from results.
                ret[f"{limb}-{data[0][3]}"] = [side, 0, 0]
            else:
                ret[f"{limb}-{data[0][3]}"] = [side, start_idx - 1 if start_idx > 0 else start_idx, end_idx + 1]
        else:
            # This will only work if we have at least 3 contacts for the trajectory - this should be the case.
            # This checks the other limb whether trailing or leading, so that we make a swap.
            data = foot_contact_order[f"{foot.name[:2]}{'L' if side == 'right' else 'R'}_foot"]
            ret[f"{limb}-{'leading' if data[0][3] == 'trailing' else 'trailing'}"] = [side, 0, 0]
    return ret


def gait_analysis(robot: pe.system.System3D, contacts):
    nfe = len(robot.m.fe)
    power_arr = get_power_values(robot, 1)
    data = {
        "angle": {},
        "torque": {},
        "power": {},
        "hindlimb-leading-y-indices":
        np.arange(contacts["hindlimb-leading"][1], contacts["hindlimb-leading"][2]),
        "hindlimb-trailing-y-indices":
        np.arange(contacts["hindlimb-trailing"][1], contacts["hindlimb-trailing"][2]),
        "hindlimb-leading-x-indices":
        np.linspace(0, 100, contacts["hindlimb-leading"][2] - contacts["hindlimb-leading"][1]),
        "hindlimb-trailing-x-indices":
        np.linspace(0, 100, contacts["hindlimb-trailing"][2] - contacts["hindlimb-trailing"][1]),
        "forelimb-leading-y-indices":
        np.arange(contacts["forelimb-leading"][1], contacts["forelimb-leading"][2]),
        "forelimb-trailing-y-indices":
        np.arange(contacts["forelimb-trailing"][1], contacts["forelimb-trailing"][2]),
        "forelimb-leading-x-indices":
        np.linspace(0, 100, contacts["forelimb-leading"][2] - contacts["forelimb-leading"][1]),
        "forelimb-trailing-x-indices":
        np.linspace(0, 100, contacts["forelimb-trailing"][2] - contacts["forelimb-trailing"][1]),
    }
    total_mass = sum(link.mass for link in robot.links)
    scale_forces_by = total_mass * 9.81
    power = {}
    torque = {}
    for motor, _power in zip(pe.motor.torques(robot), power_arr):
        power[motor.name] = _power
        torque[motor.name] = np.array([
            scale_forces_by * tau.value
            for tau in [motor.pyomo_vars["Tc"][fe, idx] for fe in robot.m.fe for idx in motor.pyomo_sets["Tc_set"]]
        ]).reshape((nfe, -1))

    body_F = pe.utils.get_vals(robot["bodyF"].pyomo_vars["q"], (robot["bodyF"].pyomo_sets["q_set"], )).squeeze(axis=1)
    body_B = pe.utils.get_vals(robot["base"].pyomo_vars["q"], (robot["base"].pyomo_sets["q_set"], )).squeeze(axis=1)
    body_B = body_B[:, 3:]
    calf_fr = pe.utils.get_vals(robot["LFR"].pyomo_vars["q"], (robot["LFR"].pyomo_sets["q_set"], )).squeeze(axis=1)
    calf_fl = pe.utils.get_vals(robot["LFL"].pyomo_vars["q"], (robot["LFL"].pyomo_sets["q_set"], )).squeeze(axis=1)
    calf_br = pe.utils.get_vals(robot["LBR"].pyomo_vars["q"], (robot["LBR"].pyomo_sets["q_set"], )).squeeze(axis=1)
    calf_bl = pe.utils.get_vals(robot["LBL"].pyomo_vars["q"], (robot["LBL"].pyomo_sets["q_set"], )).squeeze(axis=1)
    thigh_fr = body_F - pe.utils.get_vals(robot["UFR"].pyomo_vars["q"],
                                          (robot["UFR"].pyomo_sets["q_set"], )).squeeze(axis=1)
    thigh_fl = body_F - pe.utils.get_vals(robot["UFL"].pyomo_vars["q"],
                                          (robot["UFL"].pyomo_sets["q_set"], )).squeeze(axis=1)
    thigh_br = body_B - pe.utils.get_vals(robot["UBR"].pyomo_vars["q"],
                                          (robot["UBR"].pyomo_sets["q_set"], )).squeeze(axis=1)
    thigh_bl = body_B - pe.utils.get_vals(robot["UBL"].pyomo_vars["q"],
                                          (robot["UBL"].pyomo_sets["q_set"], )).squeeze(axis=1)
    hock_fr = calf_fr - pe.utils.get_vals(robot["HFR"].pyomo_vars["q"],
                                          (robot["HFR"].pyomo_sets["q_set"], )).squeeze(axis=1)
    hock_fl = calf_fl - pe.utils.get_vals(robot["HFL"].pyomo_vars["q"],
                                          (robot["HFL"].pyomo_sets["q_set"], )).squeeze(axis=1)
    hock_br = calf_br - pe.utils.get_vals(robot["HBR"].pyomo_vars["q"],
                                          (robot["HBR"].pyomo_sets["q_set"], )).squeeze(axis=1)
    hock_bl = calf_bl - pe.utils.get_vals(robot["HBL"].pyomo_vars["q"],
                                          (robot["HBL"].pyomo_sets["q_set"], )).squeeze(axis=1)

    for key, values in contacts.items():
        if values[0] == "right" and "hindlimb" in key:
            data["angle"][f"{key}-hip"] = np.degrees(thigh_br[data[f"{key}-y-indices"], 1])
            data["angle"][f"{key}-hock"] = np.degrees(hock_br[data[f"{key}-y-indices"], 1])
            data["torque"][f"{key}-hip"] = torque["back-right-hip-pitch"][data[f"{key}-y-indices"]]
            data["torque"][f"{key}-hock"] = torque["LBR_HBR_torque"][data[f"{key}-y-indices"]]
            data["power"][f"{key}-hip"] = power["back-right-hip-pitch"][data[f"{key}-y-indices"]]
            data["power"][f"{key}-hock"] = power["LBR_HBR_torque"][data[f"{key}-y-indices"]]
        elif values[0] == "right" and "forelimb" in key:
            data["angle"][f"{key}-hip"] = np.degrees(thigh_fr[data[f"{key}-y-indices"], 1])
            data["angle"][f"{key}-hock"] = np.degrees(hock_fr[data[f"{key}-y-indices"], 1])
            data["torque"][f"{key}-hip"] = torque["front-right-hip-pitch"][data[f"{key}-y-indices"]]
            data["torque"][f"{key}-hock"] = torque["LFR_HFR_torque"][data[f"{key}-y-indices"]]
            data["power"][f"{key}-hip"] = power["front-right-hip-pitch"][data[f"{key}-y-indices"]]
            data["power"][f"{key}-hock"] = power["LFR_HFR_torque"][data[f"{key}-y-indices"]]
        elif values[0] == "left" and "hindlimb" in key:
            data["angle"][f"{key}-hip"] = np.degrees(thigh_bl[data[f"{key}-y-indices"], 1])
            data["angle"][f"{key}-hock"] = np.degrees(hock_bl[data[f"{key}-y-indices"], 1])
            data["torque"][f"{key}-hip"] = torque["back-left-hip-pitch"][data[f"{key}-y-indices"]]
            data["torque"][f"{key}-hock"] = torque["LBL_HBL_torque"][data[f"{key}-y-indices"]]
            data["power"][f"{key}-hip"] = power["back-left-hip-pitch"][data[f"{key}-y-indices"]]
            data["power"][f"{key}-hock"] = power["LBL_HBL_torque"][data[f"{key}-y-indices"]]
        elif values[0] == "left" and "forelimb" in key:
            data["angle"][f"{key}-hip"] = np.degrees(thigh_fl[data[f"{key}-y-indices"], 1])
            data["angle"][f"{key}-hock"] = np.degrees(hock_fl[data[f"{key}-y-indices"], 1])
            data["torque"][f"{key}-hip"] = torque["front-left-hip-pitch"][data[f"{key}-y-indices"]]
            data["torque"][f"{key}-hock"] = torque["LFL_HFL_torque"][data[f"{key}-y-indices"]]
            data["power"][f"{key}-hip"] = power["front-left-hip-pitch"][data[f"{key}-y-indices"]]
            data["power"][f"{key}-hock"] = power["LFL_HFL_torque"][data[f"{key}-y-indices"]]

    return data


def plot_gait_attributes(results: List[Dict], dir_prefix: str):
    import matplotlib.pyplot as plt
    data_idx = ("hindlimb-leading-hip", "hindlimb-trailing-hip", "forelimb-leading-shoulder",
                "forelimb-trailing-shoulder", "hindlimb-leading-hock", "hindlimb-trailing-hock",
                "forelimb-leading-carpus", "forelimb-trailing-carpus")
    for i, titles in enumerate((("Shoulder Joints", "Front Ankle Joints"), ("Hip Joints", "Back Ankle Joints"))):
        fig, axs = plt.subplots(2, 2, figsize=(16, 9), dpi=120)
        axs = axs.ravel()
        for data in results:
            if "Shoulder" in titles[0]:
                data_idx = "forelimb"
            else:
                data_idx = "hindlimb"

            axs[0].plot(data[f"{data_idx}-leading-x-indices"],
                        data["angle"][f"{data_idx}-leading-hip"],
                        label="Leading",
                        color=misc.plot_color["green"])
            axs[0].plot(data[f"{data_idx}-trailing-x-indices"],
                        data["angle"][f"{data_idx}-trailing-hip"],
                        label="Trailing",
                        color=misc.plot_color["orange"])
            axs[0].set_ylabel('Angle ($\degree$)')
            axs[0].set_title(titles[0])
            axs[2].plot(data[f"{data_idx}-leading-x-indices"],
                        data["torque"][f"{data_idx}-leading-hip"],
                        label="Leading",
                        color=misc.plot_color["green"])
            axs[2].plot(data[f"{data_idx}-trailing-x-indices"],
                        data["torque"][f"{data_idx}-trailing-hip"],
                        label="Trailing",
                        color=misc.plot_color["orange"])
            axs[2].set_ylabel('Torque (Nm)')
            axs[1].plot(data[f"{data_idx}-leading-x-indices"],
                        data["angle"][f"{data_idx}-leading-hock"],
                        label="Leading",
                        color=misc.plot_color["green"])
            axs[1].plot(data[f"{data_idx}-trailing-x-indices"],
                        data["angle"][f"{data_idx}-trailing-hock"],
                        label="Trailing",
                        color=misc.plot_color["orange"])
            axs[1].set_title(titles[1])
            axs[3].plot(data[f"{data_idx}-leading-x-indices"],
                        data["torque"][f"{data_idx}-leading-hock"],
                        label="Leading",
                        color=misc.plot_color["green"])
            axs[3].plot(data[f"{data_idx}-trailing-x-indices"],
                        data["torque"][f"{data_idx}-trailing-hock"],
                        label="Trailing",
                        color=misc.plot_color["orange"])
            # axs[4].plot(data[f"{data_idx}-leading-x-indices"],
            #             data["power"][f"{data_idx}-leading-hip"],
            #             label="Leading",
            #             color=misc.plot_color["blue"])
            # axs[4].plot(data[f"{data_idx}-trailing-x-indices"],
            #             data["power"][f"{data_idx}-trailing-hip"],
            #             label="Trailing",
            #             color=misc.plot_color["red"])
            # axs[4].set_ylabel('Power (Nm/kg)')
            # axs[4].set_xlabel('$\%$ Stance')
            # axs[5].plot(data[f"{data_idx}-leading-x-indices"],
            #             data["power"][f"{data_idx}-leading-hock"],
            #             label="Leading",
            #             color=misc.plot_color["blue"])
            # axs[5].plot(data[f"{data_idx}-trailing-x-indices"],
            #             data["power"][f"{data_idx}-trailing-hock"],
            #             label="Trailing",
            #             color=misc.plot_color["red"])
            # axs[5].set_xlabel('$\%$ Stance')
            axs[3].set_xlabel('$\%$ Stance')
            axs[2].set_xlabel('$\%$ Stance')
        # for ax in axs:
        axs[0].legend(("Leading", "Trailing"))
        # for i in contacts:
        #     ax.axvspan(time_steps[i[0]], time_steps[i[1] - 1], facecolor='0.2', alpha=0.2)
        plt.savefig(os.path.join(
            os.path.join(dir_prefix, f"{'forelimb-analysis' if i == 0 else 'hindlimb-analysis'}.pdf")),
                    bbox_inches="tight")
        plt.close()


def get_positions_from_pose(estimator: opt.CheetahEstimator, q: np.ndarray):
    robot = estimator.model
    m = robot.m
    markers = misc.get_markers()
    link_lengths = []
    for link in robot.links:
        link_lengths.append(link.length)
    # link_lengths = np.tile(link_lengths, q.shape[0]).reshape(q.shape[0], -1)
    states = q.tolist() + link_lengths
    return [[
        estimator.position_funcs[l][0](states), estimator.position_funcs[l][1](states),
        estimator.position_funcs[l][2](states)
    ] for l in range(len(markers))]


def plot_3d_pose(estimator: opt.CheetahEstimator, pose_idx: int, display_good_pose: bool = True, view_angle=(20, 135)):
    import matplotlib.pyplot as plt
    fte_states = data_ops.load_pickle(os.path.join(estimator.params.data_dir, "fte_kinematic", "fte.pickle"))
    # Get pose from solution and perform a transformation on the root orientation to show difference in likelihood.
    pos1 = fte_states["positions"][pose_idx, :, :]
    x_orig = fte_states["x"][pose_idx, :]
    q_orig = fte_states["q"][pose_idx, :]
    pose_gmm = PoseModelGMM(os.path.join(
        estimator.params.data_dir.split("cheetah_videos")[0], "cheetah_runs", "v6", "model", "dataset_full_pose.h5"),
                            num_vars=28,
                            ext_dim=6,
                            n_comps=5)
    q_transformed = q_orig.copy()
    q_transformed[3:12:3] = np.pi / 6
    q_transformed[3:12:2] = -np.pi / 6
    x_transformed = np.array(pe.utils.flatten(misc.get_relative_angles(q_transformed.reshape(1, -1),
                                                                       0)))[misc.get_relative_angle_mask()]
    pos2 = np.array(get_positions_from_pose(estimator, q_transformed))
    print(f"Original pose log-likelihood: {pose_gmm.gmm.score(x_orig[6:].reshape(1, -1)):.2f}")
    print(f"Transformed pose log-likelihood: {pose_gmm.gmm.score(x_transformed[6:].reshape(1, -1)):.2f}")
    # Center the pose.
    pos1 -= np.expand_dims(np.mean(pos1, axis=0), axis=0)
    pos2 -= np.expand_dims(np.mean(pos2, axis=0), axis=0)
    # Plot the pose.
    fig = plt.figure(figsize=(6, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    lines_idxs = [
        [0, 1, "charcoal"],
        [0, 2, "charcoal"],
        [1, 2, "charcoal"],
        [1, 3, "charcoal"],
        [0, 3, "charcoal"],
        [2, 3, "charcoal"],
        [3, 4, "charcoal"],
        [4, 5, "charcoal"],
        [5, 6, "charcoal"],
        [6, 7, "charcoal"],
        [3, 8, "charcoal"],
        [4, 8, "charcoal"],
        [8, 9, "red"],
        [9, 10, "red"],
        [10, 11, "red"],  # right front leg
        [3, 12, "charcoal"],
        [4, 12, "charcoal"],
        [12, 13, "green"],
        [13, 14, "green"],
        [14, 15, "green"],  # left front leg
        [4, 16, "charcoal"],
        [5, 16, "charcoal"],
        [16, 17, "red"],
        [17, 18, "red"],
        [18, 19, "red"],  # right back leg
        [4, 20, "charcoal"],
        [5, 20, "charcoal"],
        [20, 21, "green"],
        [21, 22, "green"],
        [22, 23, "green"]  # left back leg
    ]
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    for line in lines_idxs:
        if display_good_pose:
            ax.plot(
                [pos1[line[0], 0], pos1[line[1], 0]],
                [pos1[line[0], 1], pos1[line[1], 1]],
                [pos1[line[0], 2], pos1[line[1], 2]],
                color=misc.plot_color[line[2]],
                # marker=".",
                # markersize=20.0,
                linewidth=3.0)
        else:
            ax.plot(
                [pos2[line[0], 0], pos2[line[1], 0]],
                [pos2[line[0], 1], pos2[line[1], 1]],
                [pos2[line[0], 2], pos2[line[1], 2]],
                color=misc.plot_color[line[2]],
                # alpha=0.4,
                # marker=".",
                # markersize=20.0,
                linewidth=3.0)
    ax.view_init(*view_angle)
    # ax.set_title(titles[i])
    ax.set_xlabel("X-axis (m)")
    ax.set_ylabel("Y-axis (m)")
    ax.set_zlabel("Z-axis (m)")
    ax.set_xlim((-1.0, 1.0))
    ax.set_ylim((-0.5, 0.5))
    ax.set_zlim((-0.5, 0.5))
    ax.grid(False)
    plt.show(block=False)
    fig.savefig(os.path.join(estimator.params.data_dir, f"pose-example-{'good' if display_good_pose else 'bad'}.pdf"),
                bbox_inches="tight",
                pad_inches=0)


def plot_eom_error(estimator: opt.CheetahEstimator):
    import matplotlib.pyplot as plt
    robot = estimator.model
    eom_err = pe.utils.get_vals_v(robot.m.slack_eom, (robot.m.fe, robot.m.cp, range(len(robot.eom)))).squeeze()
    nfe = len(robot.m.fe)
    total_time = pe.utils.total_time(robot.m)
    time_steps = np.linspace(0, total_time, num=nfe)
    q, _, _ = robot.get_state_vars()
    q_state = pe.utils.flatten(q.tolist())
    fig = plt.figure(figsize=(18, 64), dpi=120)
    axs = fig.subplots(18, 3)
    axs = axs.flatten()
    for i, q_val in enumerate(q_state):
        axs[i].plot(time_steps, eom_err[:, i], color=misc.plot_color["orange"])
        axs[i].set_title(f"{q_val} (rmse: {np.sqrt(np.mean(eom_err[:, i]**2)):.6f})")
    plt.show(block=False)
    plt.close()


def ablation_study(dir_prefix: str, result_type: Literal["data-driven", "physics-based"]):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    data1 = data_ops.load_pickle(os.path.join(dir_prefix, "data_driven_ablation_study.pickle"))
    data2 = data_ops.load_pickle(os.path.join(dir_prefix, "physics_based_ablation_study.pickle"))
    scenarios = ("Default", "Pose", "Motion", "Pose + Motion")
    width = 0.25  # the width of the bars
    x = np.arange(len(scenarios))
    fig = plt.figure(figsize=(16, 9), dpi=120)
    gs = gridspec.GridSpec(2, 4)
    # gs.update(wspace=0.5)
    ax1 = plt.subplot(gs[0, :2], )
    ax2 = plt.subplot(gs[0, 2:])
    ax3 = plt.subplot(gs[1, 1:3])
    # axs = fig.subplots(2, 2)
    # axs = axs.flatten()
    rect1 = ax1.bar(x - width / 2, [data1["mpe"][0], data1["mpe"][2], data1["mpe"][1], data1["mpe"][3]],
                    width,
                    label="Data-driven")
    rect2 = ax1.bar(x + width / 2, [data2["mpe"][0], data2["mpe"][2], data2["mpe"][1], data2["mpe"][3]],
                    width,
                    label="Physics-based")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.set_ylabel("MPE (mm)")
    ax1.set_ylim((0.0, 400.0))
    # axs[0].legend()
    # axs[0].bar_label(rect1, padding=3)
    # axs[0].bar_label(rect2, padding=3)

    rect1 = ax2.bar(x - width / 2, [data1["mpjpe"][0], data1["mpjpe"][2], data1["mpjpe"][1], data1["mpjpe"][3]],
                    width,
                    label="Data-driven")
    rect2 = ax2.bar(x + width / 2, [data2["mpjpe"][0], data2["mpjpe"][2], data2["mpjpe"][1], data2["mpjpe"][3]],
                    width,
                    label="Physics-based")
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.set_ylabel("MPJPE (mm)")
    ax2.set_ylim((0.0, 150.0))
    # axs[1].legend()
    # axs[1].bar_label(rect1, padding=3)
    # axs[1].bar_label(rect2, padding=3)

    # axs[2].bar(scenaridos, [data1["centroid_vel_rmse"][0], data1["centroid_vel_rmse"][2], data1["centroid_vel_rmse"][1], data1["centroid_vel_rmse"][3]])
    # axs[2].bar(scenaridos, [data2["centroid_vel_rmse"][0], data2["centroid_vel_rmse"][2], data2["centroid_vel_rmse"][1], data2["centroid_vel_rmse"][3]])
    rect1 = ax3.bar(x - width / 2, [
        data1["centroid_vel_rmse"][0], data1["centroid_vel_rmse"][2], data1["centroid_vel_rmse"][1],
        data1["centroid_vel_rmse"][3]
    ],
                    width,
                    label="Data-driven")
    rect2 = ax3.bar(x + width / 2, [
        data2["centroid_vel_rmse"][0], data2["centroid_vel_rmse"][2], data2["centroid_vel_rmse"][1],
        data2["centroid_vel_rmse"][3]
    ],
                    width,
                    label="Physics-based")
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenarios)
    ax3.set_ylabel("CVR (m)")
    ax3.set_ylim((0.0, 2.0))
    # axs[2].legend()
    # axs[2].bar_label(rect1, padding=3)
    # axs[2].bar_label(rect2, padding=3)

    # axs[3].plot(x, data["smoothness_error"], marker="o")
    # axs[3].set_xticks(x)
    # axs[3].set_xticklabels(scenaridos)
    # axs[3].set_ylabel("Smoothness Error (mm)")
    fig.legend(("Data-driven", "Physics-based"), loc="lower right")
    fig.savefig(os.path.join(dir_prefix, "ablation-study.pdf"), bbox_inches="tight")
    # plt.show(block=False)
    plt.close()


def data_driven_analysis(dir_prefix: str):
    import matplotlib.pyplot as plt
    data = data_ops.load_pickle(os.path.join(dir_prefix, "grid_search.pickle"))
    n_comps = (1, 2, 3, 4, 5, 6, 7)
    window_size = (1, 2, 3, 4, 5, 6, 7)
    lasso = (True, False)
    data_keys = []
    for num_comp in n_comps:
        for sparse_sol in lasso:
            for w_size in window_size:
                data_keys.append((num_comp, sparse_sol, w_size))

    fig = plt.figure(figsize=(16, 9), dpi=120)
    plt.plot(n_comps, data["gmm_train_likelihood"], marker="o", label="Train")
    plt.plot(n_comps, data["gmm_validation_likelihood"], marker="o", label="Test")
    plt.xlabel("# Components")
    plt.ylabel("Likelihood")
    plt.legend()
    plt.show(block=False)
    plt.close()

    fig = plt.figure(figsize=(16, 9), dpi=120)
    axd = fig.subplot_mosaic([['left', 'right'], ['bottom', 'bottom']])
    axd["left"].plot(window_size, data["lr_train_rmse"][:len(window_size)], marker="o", label="Train")
    axd["left"].plot(window_size, data["lr_validation_rmse"][:len(window_size)], marker="o", label="Test")
    axd["left"].set_title("L1-norm")
    axd["left"].set_ylabel("Model RMSE")
    axd["left"].set_xlabel("Window Size")
    axd["left"].legend()
    axd["right"].plot(window_size, data["lr_train_rmse"][len(window_size):], marker="o", label="Train")
    axd["right"].plot(window_size, data["lr_validation_rmse"][len(window_size):], marker="o", label="Test")
    axd["right"].set_title("L2-norm")
    axd["right"].set_ylabel("Model RMSE")
    axd["right"].set_xlabel("Window Size")
    axd["right"].legend()
    axd["bottom"].plot(window_size, data["lr_non_zeros"][:len(window_size)], marker="o", label="L1-norm")
    axd["bottom"].plot(window_size, data["lr_non_zeros"][len(window_size):], marker="o", label="L2-norm")
    axd["bottom"].set_ylabel("# Non-zero Parameters")
    axd["bottom"].set_xlabel("Window Size")
    axd["bottom"].legend()
    fig.savefig(os.path.join(dir_prefix, "lr-model-selection.pdf"), bbox_inches="tight")
    plt.close()

    comp_block = len(window_size) * len(lasso)
    n_comps_vs_mpe = np.array([np.mean(data["mpe"][i:i + comp_block]) for i in range(0, len(data["mpe"]), comp_block)])
    n_comps_vs_mpe_std = np.array(
        [np.std(data["mpe"][i:i + comp_block]) for i in range(0, len(data["mpe"]), comp_block)])
    n_comps_vs_mpjpe = np.array(
        [np.mean(data["mpjpe"][i:i + comp_block]) for i in range(0, len(data["mpjpe"]), comp_block)])
    n_comps_vs_mpjpe_std = np.array(
        [np.std(data["mpjpe"][i:i + comp_block]) for i in range(0, len(data["mpjpe"]), comp_block)])
    n_comps_vs_time = np.array([
        np.mean(data["optimisation_time"][i:i + comp_block])
        for i in range(0, len(data["optimisation_time"]), comp_block)
    ])
    n_comps_vs_time_std = np.array([
        np.std(data["optimisation_time"][i:i + comp_block])
        for i in range(0, len(data["optimisation_time"]), comp_block)
    ])

    fig = plt.figure(figsize=(16, 9), dpi=120)
    axd = fig.subplot_mosaic([['left', 'right'], ['bottom', 'bottom']])
    axd["left"].plot(n_comps, n_comps_vs_mpe, marker="o")
    axd["left"].fill_between(n_comps,
                             n_comps_vs_mpe - n_comps_vs_mpe_std,
                             n_comps_vs_mpe + n_comps_vs_mpe_std,
                             alpha=0.1)
    axd["left"].set_ylabel("Global MPE (mm)")
    axd["left"].set_xlabel("# Components")
    axd["right"].plot(n_comps, n_comps_vs_mpjpe, marker="o")
    axd["right"].fill_between(n_comps,
                              n_comps_vs_mpjpe - n_comps_vs_mpjpe_std,
                              n_comps_vs_mpjpe + n_comps_vs_mpjpe_std,
                              alpha=0.1)
    axd["right"].set_ylabel("Root-relative MPJPE (mm)")
    axd["right"].set_xlabel("# Components")
    axd["bottom"].plot(n_comps, n_comps_vs_time, marker="o")
    axd["bottom"].fill_between(n_comps,
                               n_comps_vs_time - n_comps_vs_time_std,
                               n_comps_vs_time + n_comps_vs_time_std,
                               alpha=0.1)
    axd["bottom"].set_ylabel("Time (s)")
    axd["bottom"].set_xlabel("# Components")
    plt.show(block=False)
    plt.close()

    check = [data_keys[i::comp_block] for i in range(0, len(window_size))]
    window_size_vs_mpe = np.array([np.mean(data["mpe"][i::comp_block]) for i in range(0, len(window_size))])
    window_size_vs_mpe_std = np.array([np.std(data["mpe"][i::comp_block]) for i in range(0, len(window_size))])
    window_size_vs_mpjpe = np.array([np.mean(data["mpjpe"][i::comp_block]) for i in range(0, len(window_size))])
    window_size_vs_mpjpe_std = np.array([np.std(data["mpjpe"][i::comp_block]) for i in range(0, len(window_size))])
    window_size_vs_time = np.array(
        [np.mean(data["optimisation_time"][i::comp_block]) for i in range(0, len(window_size))])
    window_size_vs_time_std = np.array(
        [np.std(data["optimisation_time"][i::comp_block]) for i in range(0, len(window_size))])
    check = [data_keys[i::comp_block] for i in range(len(window_size), 2 * len(window_size))]
    window_size_vs_mpe_2 = np.array(
        [np.mean(data["mpe"][i::comp_block]) for i in range(len(window_size), 2 * len(window_size))])
    window_size_vs_mpe_2_std = np.array(
        [np.std(data["mpe"][i::comp_block]) for i in range(len(window_size), 2 * len(window_size))])
    window_size_vs_mpjpe_2 = np.array(
        [np.mean(data["mpjpe"][i::comp_block]) for i in range(len(window_size), 2 * len(window_size))])
    window_size_vs_mpjpe_2_std = np.array(
        [np.std(data["mpjpe"][i::comp_block]) for i in range(len(window_size), 2 * len(window_size))])
    window_size_vs_time_2 = np.array(
        [np.mean(data["optimisation_time"][i::comp_block]) for i in range(len(window_size), 2 * len(window_size))])
    window_size_vs_time_2_std = np.array(
        [np.std(data["optimisation_time"][i::comp_block]) for i in range(len(window_size), 2 * len(window_size))])
    fig = plt.figure(figsize=(16, 9), dpi=120)
    axd = fig.subplot_mosaic([['left', 'right'], ['bottom', 'bottom']])
    axd["left"].plot(window_size, window_size_vs_mpe, marker="o", label="L1-norm")
    axd["left"].fill_between(window_size,
                             window_size_vs_mpe - window_size_vs_mpe_std,
                             window_size_vs_mpe + window_size_vs_mpe_std,
                             alpha=0.1)
    axd["left"].plot(window_size, window_size_vs_mpe_2, marker="o", label="L2-norm")
    axd["left"].fill_between(window_size,
                             window_size_vs_mpe_2 - window_size_vs_mpe_2_std,
                             window_size_vs_mpe_2 + window_size_vs_mpe_2_std,
                             alpha=0.1)
    axd["left"].set_ylabel("Global MPE (mm)")
    axd["left"].set_xlabel("Window Size")
    axd["left"].legend()
    axd["right"].plot(window_size, window_size_vs_mpjpe, marker="o", label="L1-norm")
    axd["right"].fill_between(window_size,
                              window_size_vs_mpjpe - window_size_vs_mpjpe_std,
                              window_size_vs_mpjpe + window_size_vs_mpjpe_std,
                              alpha=0.1)
    axd["right"].plot(window_size, window_size_vs_mpjpe_2, marker="o", label="L2-norm")
    axd["right"].fill_between(window_size,
                              window_size_vs_mpjpe_2 - window_size_vs_mpjpe_2_std,
                              window_size_vs_mpjpe_2 + window_size_vs_mpjpe_2_std,
                              alpha=0.1)
    axd["right"].set_ylabel("Root-relative MPJPE (mm)")
    axd["right"].set_xlabel("Window Size")
    axd["right"].legend()
    axd["bottom"].plot(window_size, window_size_vs_time, marker="o", label="L1-norm")
    axd["bottom"].fill_between(window_size,
                               window_size_vs_time - window_size_vs_time_std,
                               window_size_vs_time + window_size_vs_time_std,
                               alpha=0.1)
    axd["bottom"].plot(window_size, window_size_vs_time_2, marker="o", label="L2-norm")
    axd["bottom"].fill_between(window_size,
                               window_size_vs_time_2 - window_size_vs_time_2_std,
                               window_size_vs_time_2 + window_size_vs_time_2_std,
                               alpha=0.1)
    axd["bottom"].set_ylabel("Time (s)")
    axd["bottom"].set_xlabel("Window Size")
    axd["bottom"].legend()
    plt.show(block=False)
    plt.close()

    type_markers = ('o', 'd')
    f = lambda m, c: ax.plot([], [], marker=m, color=c, ls="none")[0]
    t = [i[0] for i in data_keys]
    x = [i[2] for i in data_keys]
    z = [int(i[1]) for i in data_keys]
    y = data["mpe"]
    import matplotlib.colors as colors
    import matplotlib.cm as cmx
    from matplotlib.collections import LineCollection
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.subplots(1, 2)
    ax.flatten()
    cmap = plt.get_cmap('plasma', np.max(t))
    # cNorm  = colors.Normalize(vmin=0, vmax=max(t))
    cNorm = colors.BoundaryNorm(np.arange(1, np.max(t) + 2), cmap.N)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    x1, x2 = [], []
    y1, y2 = [], []
    for i in range(len(data_keys)):
        if z[i]:
            ax[0].scatter(x[i], y[i], color=scalarMap.to_rgba(t[i]), s=30, cmap=cmap, edgecolor='none')
            x1.append(x[i])
            y1.append(y[i])
        else:
            ax[1].scatter(x[i], y[i], color=scalarMap.to_rgba(t[i]), s=30, cmap=cmap, edgecolor='none')
            x2.append(x[i])
            y2.append(y[i])
    # scalarMap.set_array([])
    segs1 = np.stack((x1, y1), axis=1).reshape(7, 7, 2)
    segs2 = np.stack((x2, y2), axis=1).reshape(7, 7, 2)
    c = [scalarMap.to_rgba(i) for i in np.unique(t)]
    lc = LineCollection(segs1, colors=c, array=np.unique(t))
    ax[0].add_collection(lc)
    lc = LineCollection(segs2, colors=c, array=np.unique(t))
    ax[1].add_collection(lc)
    cbar = fig.colorbar(scalarMap, ax=ax[1])
    cbar.ax.set_ylabel("# Components", rotation=270, labelpad=25)
    tick_locs = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    cbar_tick_label = np.arange(np.min(t), np.max(t) + 1)
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(cbar_tick_label)
    # handles = [f(type_markers[i], "k") for i in range(2)]
    # ax.legend(handles, ["L2-norm", "L1-norm"], loc="center right", framealpha=1)
    # ax[0].set_xlabel("Window Size")
    ax[0].set_title("L1-norm")
    ax[1].set_title("L2-norm")
    ax[0].set_ylim((150, 310))
    ax[1].set_ylim((150, 310))
    ax[1].set_yticks([])
    # plt.xlabel("Window Size")
    # ax[0].set_ylabel("Global MPE (mm)")
    fig.text(0.5, 0.0, "Window Size", ha="center")
    fig.text(0.0, 0.5, "MPE (mm)", va="center", rotation="vertical")
    # for a in [0.1, 0.5, 0.9]:
    #     ax.scatter([], [], c='k', alpha=0.5, s=a*100, label=str(a), edgecolors='none')
    # l1 = ax.legend(scatterpoints=1, frameon=True, loc='lower left' ,markerscale=1)
    # for b in [0.25, 0.5, 0.75]:
    #     ax.scatter([], [], c='k', alpha=b, s=50, label=str(b), edgecolors='none')
    # ax.legend(scatterpoints=1, frameon=True, loc='lower right' ,markerscale=1)
    fig.savefig(os.path.join(dir_prefix, "cross-validation-mpe.pdf"), bbox_inches="tight")
    # plt.show(block=False)
    plt.close()

    y = data["mpjpe"]
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.subplots(1, 2)
    ax.flatten()
    cmap = plt.get_cmap('plasma', np.max(t))
    # cNorm  = colors.Normalize(vmin=0, vmax=max(t))
    cNorm = colors.BoundaryNorm(np.arange(1, np.max(t) + 2), cmap.N)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    x1, x2 = [], []
    y1, y2 = [], []
    for i in range(len(data_keys)):
        if z[i]:
            ax[0].scatter(x[i], y[i], color=scalarMap.to_rgba(t[i]), s=30, cmap=cmap, edgecolor='none')
            x1.append(x[i])
            y1.append(y[i])
        else:
            ax[1].scatter(x[i], y[i], color=scalarMap.to_rgba(t[i]), s=30, cmap=cmap, edgecolor='none')
            x2.append(x[i])
            y2.append(y[i])
    # scalarMap.set_array([])
    segs1 = np.stack((x1, y1), axis=1).reshape(7, 7, 2)
    segs2 = np.stack((x2, y2), axis=1).reshape(7, 7, 2)
    c = [scalarMap.to_rgba(i) for i in np.unique(t)]
    lc = LineCollection(segs1, colors=c, array=np.unique(t))
    ax[0].add_collection(lc)
    lc = LineCollection(segs2, colors=c, array=np.unique(t))
    ax[1].add_collection(lc)
    cbar = fig.colorbar(scalarMap, ax=ax[1])
    cbar.ax.set_ylabel("# Components", rotation=270, labelpad=25)
    tick_locs = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]
    cbar_tick_label = np.arange(np.min(t), np.max(t) + 1)
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(cbar_tick_label)

    # ax.set_xlabel("Window Size")
    # ax.set_ylabel("Root-relative MPJPE (mm)")
    # handles = [f(type_markers[i], "k") for i in range(2)]
    # ax.legend(handles, ["L2-norm", "L1-norm"], framealpha=1)
    ax[0].set_title("L1-norm")
    ax[1].set_title("L2-norm")
    ax[0].set_ylim((75, 120))
    ax[1].set_ylim((75, 120))
    ax[1].set_yticks([])
    fig.text(0.5, 0.0, "Window Size", ha="center")
    fig.text(0.0, 0.5, "MPJPE (mm)", va="center", rotation="vertical")
    fig.savefig(os.path.join(dir_prefix, "cross-validation-mpjpe.pdf"), bbox_inches="tight")
    # plt.show(block=False)
    plt.close()


def plot_power_values(robot: pe.system.System3D):
    import matplotlib.pyplot as plt
    nfe = len(robot.m.fe)
    power_arr = get_power_values(robot, 1)
    power = {}
    torque = {}
    for motor, _power in zip(pe.motor.torques(robot), power_arr):
        power[motor.name] = _power
        torque[motor.name] = np.array([
            tau.value
            for tau in [motor.pyomo_vars["Tc"][fe, idx] for fe in robot.m.fe for idx in motor.pyomo_sets["Tc_set"]]
        ]).reshape((nfe, -1))
    peaks = np.sum(np.hstack(power_arr), axis=1)
    mean_pwr = [np.mean(peaks)] * nfe
    total_time = pe.utils.total_time(robot.m)
    time_steps = np.linspace(0, total_time, num=nfe)
    fig = plt.figure(figsize=(16, 9), dpi=120)
    plt.plot(time_steps, peaks, color=misc.plot_color["blue"])
    plt.plot(time_steps, mean_pwr, color=misc.plot_color["orange"], label='Mean', linestyle='--')
    plt.title(
        f'Total power output of cheetah.\nPeak power: {int(np.max(peaks))} W/kg, Avg power: {int(np.mean(peaks))} W/kg')
    plt.ylabel('Total power (W/kg)')
    plt.xlabel('Time (s)')
    plt.show()
    plt.close()


def plot_torques(robot: pe.system.System3D, out_dir: str):
    import matplotlib.pyplot as plt
    from cycler import cycler
    nfe = len(robot.m.fe)
    titles = ("Tail base", "Left back hip", "Right back hip", "Spine", "Left front hip", "Right front hip", "Neck",
              "Tail mid", "Left front knee", "Left front ankle", "Right front knee", "Right front ankle",
              "Left back knee", "Left back ankle", "Right back knee", "Right back ankle")
    torque = {}
    for motor in pe.motor.torques(robot):
        torque[motor.name] = []
        for idx in motor.pyomo_sets["Tc_set"]:
            torque[motor.name].append([tau.value for tau in [motor.pyomo_vars["Tc"][fe, idx] for fe in robot.m.fe]])
    total_time = pe.utils.total_time(robot.m)
    time_steps = np.linspace(0, total_time, num=nfe)
    fig = plt.figure(figsize=(12, 32), dpi=120)
    axs = fig.subplots(8, 2)
    axs = axs.flatten()
    for i, motor in enumerate(pe.motor.torques(robot)):
        axs[i].set_prop_cycle(
            cycler('color', [misc.plot_color["orange"], misc.plot_color["green"], misc.plot_color["red"]]))
        axs[i].plot(time_steps, np.array(torque[motor.name]).T)
        axs[i].set_title(titles[i])
    fig.savefig(os.path.join(out_dir, "torque-profile.pdf"), bbox_inches="tight")
    plt.close()


def torque_error(robot1: pe.system.System3D, robot2: pe.system.System3D):
    tau1 = np.empty([])
    tau2 = np.empty([])
    for motor in pe.motor.torques(robot1):
        if tau1.shape == ():
            tau1 = pe.utils.get_vals(motor.pyomo_vars["Tc"], (motor.pyomo_sets["Tc_set"], ))
        else:
            tau1 = np.concatenate((tau1, pe.utils.get_vals(motor.pyomo_vars["Tc"], (motor.pyomo_sets["Tc_set"], ))),
                                  axis=1)
    for motor in pe.motor.torques(robot2):
        if tau2.shape == ():
            tau2 = pe.utils.get_vals(motor.pyomo_vars["Tc"], (motor.pyomo_sets["Tc_set"], ))
        else:
            tau2 = np.concatenate((tau2, pe.utils.get_vals(motor.pyomo_vars["Tc"], (motor.pyomo_sets["Tc_set"], ))),
                                  axis=1)

    print(f"RMSE: {misc.rmse(tau1.flatten(), tau2.flatten())}")
    return (np.linalg.norm(tau1 - tau2, axis=0) / np.sqrt(len(tau1))), tau1, tau2


def plot_grf(robot: pe.system.System3D):
    import matplotlib.pyplot as plt
    # Scaling forces by body weight.
    # total_mass = sum(link.mass for link in robot.links)
    # force_scale = total_mass * 9.81
    GRFz = {}
    for foot in pe.foot.feet(robot):
        GRFz[foot.name] = pe.utils.get_vals(foot.pyomo_vars["GRFz"], tuple())

    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig_idx = 1
    nfe = len(robot.m.fe)
    total_time = pe.utils.total_time(robot.m)
    time_steps = np.linspace(0, total_time, num=nfe)
    result_pred = np.zeros((nfe, 1))
    for name, values in GRFz.items():
        ax = plt.subplot(2, 2, fig_idx)
        ax.plot(time_steps, values, color=misc.plot_color["orange"], label=name)
        result_pred += values
        ax.legend()
        if fig_idx == 1 or fig_idx == 3:
            ax.set_ylabel("Normalised Force (N)")
        if fig_idx == 3 or fig_idx == 4:
            ax.set_xlabel("Time (s)")
        ax.set_ylim((-0.5, 5.0))
        fig_idx += 1
    plt.show(block=False)
    plt.close()
    fig.savefig("./data/grf-estimation-subplots.pdf")
    fig = plt.figure(figsize=(16, 9), dpi=120)
    plt.plot(time_steps, result_pred, color=misc.plot_color["orange"])
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N/Body Weight)")
    plt.show()
    fig.savefig("./data/grf-estimation.pdf")


def grf_error(robot: pe.system.System3D, data_path: str, out_dir: str):
    # Scaling forces by body weight.
    total_mass = sum(link.mass for link in robot.links)
    print(f"Total cheetah mass: {total_mass:.2f} Kg")
    scale_forces_by = 1 / (total_mass * 9.81)
    grf_df = pd.read_hdf(os.path.join(data_path, "grf", "data.h5"))
    with open(os.path.join(data_path, "metadata.json"), "r") as f:
        metadata = json.load(f)
    start_frame = metadata["start_frame"]
    foot_contact_order = metadata["contacts"]
    gt_grf = {}
    for foot in pe.foot.feet(robot):
        gt_grf[foot.name] = [0.0] * len(robot.m.fe)
        # Check if the foot is in contact sequence.
        if foot.name in foot_contact_order and foot_contact_order[foot.name] is not None:
            grf = grf_df.query(f"force_plate == {foot_contact_order[foot.name][0][2]-1}")
            # Resample data from 3500Hz to 200Hz i.e 2/35 factor.
            Fz = signal.resample_poly(misc.remove_dc_offset(grf["Fz"].values, 500), up=2, down=35,
                                      axis=0) * scale_forces_by
            foot_on_ground_indices = []
            for contact_seq in foot_contact_order[foot.name]:
                foot_on_ground_indices = foot_on_ground_indices + list(range(contact_seq[0], contact_seq[1] + 1))
            for fe, cp in robot.indices(one_based=True):
                if (start_frame + fe - 1) in foot_on_ground_indices:
                    z_comp = Fz[start_frame + fe - 1]
                    if z_comp > 0:
                        gt_grf[foot.name][fe - 1] = z_comp
    # GRF error.
    _plot_grf(robot, gt_grf, out_dir)


def kinematic_error(robot: pe.system.System3D, root_dir: str, data_path: str):
    fte_states = data_ops.load_pickle(os.path.join(root_dir, data_path, "fte_kinematic", "fte.pickle"))
    fte_states2 = data_ops.load_pickle(os.path.join(root_dir, data_path, "fte_kinetic", "fte.pickle"))
    init_q = fte_states["x"]
    q_optimised = fte_states2["x"]
    print(f"RMSE base (m): {misc.rmse(init_q[:, :6], q_optimised[:, :6]):.4f}")
    print(f"RMSE links (deg): {np.degrees(misc.rmse(init_q[:, 6:], q_optimised[:, 6:])):.4f}")


def metrics(
    root_dir: str,
    data_path: str,
    start_frame: int,
    end_frame: int,
    dlc_thresh: float = 0.5,
    type_3D_gt: str = "fte",
    out_dir_prefix: str = None,
) -> Tuple[float, float, float]:
    """
    Generate metrics for a particular reconstruction. Note, the `fte.pickle` needs to be generated prior to calling this function.
    Args:
        root_dir: The root directory where the videos are stored.
        data_path: Path to video set of interest.
        start_frame: The start frame number. Note, this value is deducted by `-1` to compensate for `0` based indexing.
        end_frame: The end frame number.
        dlc_thresh: The DLC confidence score to filter 2D keypoints. Defaults to 0.5.
        use_3D_gt: Flag to select 3D ground truth for evaluation. Defaults to False.
        type_3D_gt: Sets the type of 3D ground truth to expect. Valid values are fte, pw_fte, sd_fte, pw_sd_fte.
        out_dir_prefix: Used to change the output directory from the root. This is often used if you have the cheetah data stored in a location and you want the output to be saved elsewhere.
    Returns:
        A tuple consisting of the mean error [px], median error [px], and PCK [%].
    """
    if out_dir_prefix:
        out_dir = os.path.join(out_dir_prefix, data_path, type_3D_gt)
    else:
        out_dir = os.path.join(root_dir, data_path, type_3D_gt)
    # load DLC data
    data_dir = os.path.join(root_dir, data_path)
    dlc_dir = os.path.join(data_dir, 'dlc')
    gt_dir = os.path.join(data_dir, 'dlc_hand_labeled')
    assert os.path.exists(dlc_dir)
    assert os.path.exists(gt_dir)

    try:
        k_arr, d_arr, r_arr, t_arr, cam_res, n_cams, scene_fpath = misc.find_scene_file(data_dir)
    except Exception:
        print('Early exit because extrinsic calibration files could not be located')
        return []
    d_arr = d_arr.reshape((-1, 4))

    dlc_points_fpaths = sorted(glob(os.path.join(dlc_dir, '*.h5')))
    assert n_cams == len(dlc_points_fpaths), f'# of dlc .h5 files != # of cams in {n_cams}_cam_scene_sba.json'

    # calculate residual error
    states = data_ops.load_pickle(os.path.join(out_dir, 'fte.pickle'))
    markers = misc.get_markers()
    data_location = data_path.split('/')
    test_data = [loc.capitalize() for loc in data_location]
    gt_name = str.join('', test_data)
    gt_name = gt_name.replace('Top', '').replace('Bottom', '')

    gt_points_fpaths = sorted(glob(os.path.join(gt_dir, '*.h5')))
    if len(gt_points_fpaths) > 0:
        points_2d_df = misc.load_dlc_points_as_df(gt_points_fpaths, verbose=False, hand_labeled=True)
    else:
        print('No ground truth labels for this test.')
        points_2d_df = misc.load_dlc_points_as_df(dlc_points_fpaths, verbose=False)
        points_2d_df = points_2d_df[points_2d_df['likelihood'] > dlc_thresh]  # ignore points with low likelihood

    positions_3ds = [np.array(states["positions"])] * n_cams
    points_3d_dfs = []
    for positions_3d in positions_3ds:
        frames = np.arange(start_frame, end_frame).reshape((-1, 1))
        n_frames = len(frames)
        points_3d = []
        for i, m in enumerate(markers):
            _pt3d = np.squeeze(positions_3d[:, i, :])
            marker_arr = np.array([m] * n_frames).reshape((-1, 1))
            _pt3d = np.hstack((frames, marker_arr, _pt3d))
            points_3d.append(_pt3d)
        points_3d_df = pd.DataFrame(
            np.vstack(points_3d),
            columns=['frame', 'marker', 'x', 'y', 'z'],
        ).astype({
            'frame': 'int64',
            'marker': 'str',
            'x': 'float64',
            'y': 'float64',
            'z': 'float64'
        })
        points_3d_dfs.append(points_3d_df)

    points_2d_df = points_2d_df.query(f"{start_frame} <= frame <= {end_frame}")
    px_errors = _residual_error(points_2d_df, points_3d_dfs, markers, (k_arr, d_arr, r_arr, t_arr, cam_res, n_cams))
    mean_error, med_error = _save_error_dists(px_errors, out_dir)

    # Calculate the per marker error and save results.
    marker_errors_2d = dict.fromkeys(markers, [])
    for i, m in enumerate(markers):
        temp_dist = []
        for k, df in px_errors.items():
            temp_dist += df.query(f'marker == "{m}"')['pixel_residual'].tolist()
        marker_errors_2d[m] = np.asarray(list(map(float, temp_dist)))
    error_df = pd.DataFrame(
        pd.DataFrame(dict([(k, pd.Series(v)) for k, v in marker_errors_2d.items()])).describe(include='all'))
    error_df.to_csv(os.path.join(out_dir, 'reprojection_results.csv'))

    return mean_error, med_error


def _residual_error(points_2d_df, points_3d_dfs, markers, camera_params) -> Dict:
    k_arr, d_arr, r_arr, t_arr, _, _ = camera_params
    n_cam = len(k_arr)
    if not isinstance(points_3d_dfs, list):
        points_3d_dfs = [points_3d_dfs] * n_cam
    error = {str(i): None for i in range(n_cam)}
    for i in range(n_cam):
        error_dfs = []
        for m in markers:
            # extract frames
            q = f'marker == "{m}"'
            pts_2d_df = points_2d_df.query(q + f'and camera == {i}')
            pts_3d_df = points_3d_dfs[i].query(q)
            pts_2d_df = pts_2d_df[pts_2d_df[['x', 'y']].notnull().all(axis=1)]
            pts_3d_df = pts_3d_df[pts_3d_df[['x', 'y', 'z']].notnull().all(axis=1)]
            valid_frames = np.intersect1d(pts_2d_df['frame'].to_numpy(), pts_3d_df['frame'].to_numpy())
            pts_2d_df = pts_2d_df[pts_2d_df['frame'].isin(valid_frames)].sort_values(by=['frame'])
            pts_3d_df = pts_3d_df[pts_3d_df['frame'].isin(valid_frames)].sort_values(by=['frame'])

            # get 2d and reprojected points
            frames = pts_2d_df['frame'].to_numpy()
            pts_2d = pts_2d_df[['x', 'y']].to_numpy(dtype=np.float32)
            pts_3d = pts_3d_df[['x', 'y', 'z']].to_numpy(dtype=np.float32)

            if len(pts_2d) == 0 or len(pts_3d) == 0:
                continue
            prj_2d = misc.project_points(pts_3d, k_arr[i], d_arr[i], r_arr[i], t_arr[i])

            # camera distance
            cam_pos = np.squeeze(t_arr[i, :, :])
            cam_dist = np.sqrt(np.sum((pts_3d - cam_pos)**2, axis=1))

            # compare both types of points
            residual = np.sqrt(np.sum((pts_2d - prj_2d)**2, axis=1))
            error_uv = pts_2d - prj_2d

            # make the result dataframe
            marker_arr = np.array([m] * len(frames))
            error_dfs.append(
                pd.DataFrame(np.vstack((frames, marker_arr, cam_dist, residual, error_uv.T)).T,
                             columns=['frame', 'marker', 'camera_distance', 'pixel_residual', 'error_u', 'error_v']))

        error[str(i)] = pd.concat(error_dfs, ignore_index=True) if len(error_dfs) > 0 else pd.DataFrame(
            columns=['frame', 'marker', 'camera_distance', 'pixel_residual', 'error_u', 'error_v'])

    return error


def _save_error_dists(px_errors, output_dir: str) -> Tuple[float, float, float]:
    import matplotlib.pyplot as plt
    distances = []
    for k, df in px_errors.items():
        distances += df['pixel_residual'].tolist()
    distances = np.asarray(list(map(float, distances)))

    mean_error = float(np.mean(distances))
    med_error = float(np.median(distances))
    data_ops.save_pickle(os.path.join(output_dir, 'reprojection.pickle'), {
        'error': distances,
        'mean_error': mean_error,
        'med_error': med_error,
    })

    # plot the error histogram
    xlabel = 'Error [px]'
    ylabel = 'Frequency'

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(distances, bins=50)
    ax.set_title(f'Error Overview (N={len(distances)}, \u03BC={mean_error:.3f}, med={med_error:.3f})')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(os.path.join(output_dir, 'overall_error_hist.pdf'))

    hist_data = []
    labels = []
    for k, df in px_errors.items():
        i = int(k)
        e = df['pixel_residual'].tolist()
        e = list(map(float, e))
        hist_data.append(e)
        labels.append('cam{} (N={})'.format(i + 1, len(e)))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(hist_data, bins=10, density=True, histtype='bar')
    ax.legend(labels)
    ax.set_title('Reprojection Pixel Error')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(os.path.join(output_dir, 'cams_error_hist.pdf'))

    return mean_error, med_error


def _plot_grf(robot: pe.system.System3D, gt_grf: dict, out_dir: str):
    import matplotlib.pyplot as plt
    from cycler import cycler
    # Scaling forces by body weight.
    # total_mass = sum(link.mass for link in robot.links)
    # force_scale = total_mass * 9.81
    GRFz = {}
    for foot in pe.foot.feet(robot):
        GRFz[foot.name] = pe.utils.get_vals(foot.pyomo_vars["GRFz"], tuple())

    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig_idx = 1
    total_target = []
    total_predicted = []
    nfe = len(robot.m.fe)
    total_time = pe.utils.total_time(robot.m)
    time_steps = np.linspace(0, total_time, num=nfe)
    result_gt = np.zeros((nfe, 1))
    result_pred = np.zeros((nfe, 1))
    for name, values in GRFz.items():
        ax = plt.subplot(2, 2, fig_idx)
        ax.set_prop_cycle(cycler('color', [misc.plot_color["charcoal"], misc.plot_color["gray"]]))
        ax.plot(time_steps, gt_grf[name], label=f"{name}_gt")
        ax.set_prop_cycle(cycler('color', [misc.plot_color["orange"], misc.plot_color["blue"]]))
        ax.plot(time_steps, values, label=name)
        total_target += gt_grf[name]
        total_predicted += values.flatten().tolist()
        result_gt += np.array(gt_grf[name]).reshape((nfe, -1))
        result_pred += values
        ax.legend()
        # ax.set_title(f"RMSE: {misc.rmse(values.flatten(), gt_grf[name]):.4f}")
        if fig_idx == 1 or fig_idx == 3:
            ax.set_ylabel("Normalised Force (N)")
        if fig_idx == 3 or fig_idx == 4:
            ax.set_xlabel("Time (s)")

        ax.set_ylim((-0.5, 3.0))
        fig_idx += 1
    # plt.show(block=False)
    fig.savefig(os.path.join(out_dir, "grf-estimation-subplots.pdf"), bbox_inches="tight")
    plt.close()
    fig = plt.figure(figsize=(16, 9), dpi=120)
    plt.plot(time_steps, result_gt, color=misc.plot_color["charcoal"], label="Ground truth")
    plt.plot(time_steps, result_pred, color=misc.plot_color["orange"], label="Estimated")
    plt.legend()
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N/Body Weight)")
    # plt.show()
    fig.savefig(os.path.join(out_dir, "grf-estimation.pdf"), bbox_inches="tight")
    total_weight = sum(link.mass for link in robot.links) * 9.81
    print(f"{out_dir} GRF RMSE (N): {misc.rmse(np.array(total_predicted), np.array(total_target))*total_weight}")
