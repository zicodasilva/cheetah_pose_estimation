import os
from argparse import ArgumentParser
import gc
from itertools import chain
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import cv2 as cv
import matplotlib.pyplot as plt
import json
from glob import glob
from typing import List, Optional, Tuple
from time import time
from tqdm import tqdm
import acinoset_opt as opt
import acinoset_misc as misc
from common.py_utils import data_ops


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


def validate_dataset(root_dir: str) -> List:
    markers = misc.get_markers()
    coords = ["x", "y", "z"]

    # get velocity of virtual body
    def get_velocity(position, h):
        # body COM approximated as mean of tail base, spine and neck points
        body_x = np.mean([position[m, "x"] for m in ["tail_base", "spine", "neck_base"]], 0)
        body_y = np.mean([position[m, "y"] for m in ["tail_base", "spine", "neck_base"]], 0)
        body_z = np.mean([position[m, "z"] for m in ["tail_base", "spine", "neck_base"]], 0)

        body_dx = (body_x[1:] - body_x[:-1]) / h
        body_dy = (body_y[1:] - body_y[:-1]) / h
        body_dz = (body_z[1:] - body_z[:-1]) / h

        body_v = np.sqrt(body_dx**2 + body_dy**2 + body_dz**2)
        return body_v

    # extract position data from individual pickle file
    def extract_data(filename):
        data = data_ops.load_pickle(filename)

        position = {}
        for m_i, m in enumerate(markers):
            for c_i, c in enumerate(coords):
                position.update(
                    {(m, c): np.array([data["positions"][f][m_i][c_i] for f in range(len(data["positions"]))])}
                )
        # flip direction if running towards -x
        if position["neck_base", "x"][-1] < position["neck_base", "x"][0]:
            for m in markers:
                for c in ["x", "y"]:
                    position[m, c] = -position[m, c]
        return position

    bad_trajectories = []
    fte_fpaths = sorted(glob(os.path.join(root_dir, "**/fte.pickle"), recursive=True))
    for fpath in fte_fpaths:
        position = extract_data(fpath)
        temp = fpath.split(root_dir)[1]
        info = temp.split(os.sep)
        date = info[1]
        # sanity checks
        fail = 0
        h = 1 / 90
        if date[:4] == "2017":
            h = 1 / 90
        if date[:4] == "2019":
            h = 1 / 120
        body_v = get_velocity(position, h)
        if np.max(np.abs(body_v)) > 50:  # if velocity implausibly high
            fail += 1
        for m in markers:
            if np.min(position[m, "z"]) < -0.3:  # goes too deep
                fail += 1
            if m not in ["tail_base", "tail1", "tail2"] and np.max(position[m, "z"]) > 1:  # goes too high
                fail += 1

        if fail != 0:
            bad_trajectories.append(os.sep.join(info[1:-2]))

    return bad_trajectories


def ray_from_img(img_pts, K, D):
    pts = img_pts.reshape((-1, 1, 2))
    norm_pts = cv.fisheye.undistortPoints(pts, K, D)
    norm_pts = norm_pts.reshape((-1, 2))
    norm_pts = np.append(norm_pts, [[1] for i in range(len(norm_pts))], 1).T
    return norm_pts


def distance_from_camera(data_path: str, com_pos: np.ndarray, cam_idx: int):
    # FOV of 118.2◦ in the horizontal direction and 69.5◦
    k_arr, d_arr, r_arr, t_arr, cam_res, _, _ = misc.find_scene_file(data_path)
    # center_img = np.array([k_arr[0, 2], k_arr[1, 2]])
    center_img = np.array([cam_res[0] / 2, cam_res[1] / 2])
    img_pts = misc.project_points_fisheye(com_pos, k_arr[cam_idx], d_arr[cam_idx], r_arr[cam_idx], t_arr[cam_idx])
    r1 = ray_from_img(center_img, k_arr[cam_idx], d_arr[cam_idx])
    r2 = ray_from_img(img_pts, k_arr[cam_idx], d_arr[cam_idx])
    r1 = np.tile(r1, (1, com_pos.shape[0]))
    obj_angle = []
    for i in range(com_pos.shape[0]):
        cos_angle = r1[:, i].dot(r2[:, i]) / (np.linalg.norm(r1[:, i]) * np.linalg.norm(r2[:, i]))
        obj_angle.append(np.degrees(np.arccos(cos_angle)))
    cam_pos = -np.linalg.inv(r_arr[cam_idx]) @ t_arr[cam_idx]
    return np.linalg.norm(np.tile(cam_pos.T, (com_pos.shape[0], 1)) - com_pos, axis=1), obj_angle


def align_error_trajectories(x):
    # Find the maximum length among the trajectories
    max_length = max(len(trajectory) for trajectory in x)

    # Interpolate each trajectory to have the same length
    interpolated_trajectories = []
    for trajectory in x:
        original_indices = np.linspace(0, 1, len(trajectory))
        target_indices = np.linspace(0, 1, max_length)
        interpolation_function = interp1d(original_indices, trajectory, kind='linear')
        interpolated_trajectory = interpolation_function(target_indices)
        interpolated_trajectories.append(interpolated_trajectory)

    # Convert the list of interpolated trajectories to a numpy array
    interpolated_trajectories = np.array(interpolated_trajectories)

    # Calculate the mean and standard deviation across the trajectories
    mean_trajectory = np.mean(interpolated_trajectories, axis=0)
    std_trajectory = np.std(interpolated_trajectories, axis=0)
    median_trajectory = np.median(interpolated_trajectories, axis=0)
    lower_quantile = np.quantile(interpolated_trajectories, 0.25, axis=0)
    upper_quantile = np.quantile(interpolated_trajectories, 0.75, axis=0)
    mad_trajectory = np.median(np.abs(interpolated_trajectories - median_trajectory), axis=0)

    return max_length, interpolated_trajectories, mean_trajectory, std_trajectory, median_trajectory, lower_quantile, upper_quantile, mad_trajectory


def align_error_and_plot(x, y, z, file_name: str):
    max_length_x, x_interp, x_mean, x_std, x_med, x_lower, x_upper, x_mad = align_error_trajectories(x)
    max_length_y, y_interp, y_mean, y_std, y_med, y_lower, y_upper, y_mad = align_error_trajectories(y)
    max_length_z, z_interp, z_mean, z_std, z_med, z_lower, z_upper, z_mad = align_error_trajectories(z)

    assert max_length_x == max_length_y == max_length_z

    # Plotting the mean trajectory with standard deviation
    fig = plt.figure(figsize=(16, 12), dpi=120)
    # for trajectory in x_interp:
        # plt.plot(trajectory, color=misc.plot_color["charcoal"], alpha=0.3)
    # for trajectory in y_interp:
        # plt.plot(trajectory, color=misc.plot_color["green"], alpha=0.3)
    # for trajectory in z_interp:
        # plt.plot(trajectory, color=misc.plot_color["orange"], alpha=0.3)
    # plt.plot(x_mean, color=misc.plot_color["charcoal"], label='Default')
    plt.plot(x_med, color=misc.plot_color["charcoal"], label='Default')
    # plt.fill_between(range(max_length_x), x_mean - x_std, x_mean + x_std, color=misc.plot_color["charcoal"], alpha=0.15)
    plt.fill_between(range(max_length_x), x_med - x_mad, x_med + x_mad, color=misc.plot_color["charcoal"], alpha=0.15)
    # plt.fill_between(range(max_length_x), x_lower, x_upper, color=misc.plot_color["charcoal"], alpha=0.15)
    # plt.plot(y_mean, color=misc.plot_color["green"], label='Data-driven')
    plt.plot(y_med, color=misc.plot_color["green"], label='Data-driven')
    # plt.fill_between(range(max_length_y), y_mean - y_std, y_mean + y_std, color=misc.plot_color["green"], alpha=0.15)
    plt.fill_between(range(max_length_y), y_med - y_mad, y_med + y_mad, color=misc.plot_color["green"], alpha=0.15)
    # plt.fill_between(range(max_length_y), y_lower, y_upper, color=misc.plot_color["green"], alpha=0.15)
    # plt.plot(z_mean, color=misc.plot_color["orange"], label='Physics-based')
    plt.plot(z_med, color=misc.plot_color["orange"], label='Physics-based')
    # plt.fill_between(range(max_length_z), z_mean - z_std, z_mean + z_std, color=misc.plot_color["orange"], alpha=0.15)
    plt.fill_between(range(max_length_z), z_med - z_mad, z_med + z_mad, color=misc.plot_color["orange"], alpha=0.15)
    # plt.fill_between(range(max_length_z), z_lower, z_upper, color=misc.plot_color["orange"], alpha=0.15)
    plt.title("MPE over time")
    plt.xlabel("Frames")
    plt.ylabel("Error (mm)")
    plt.legend()
    fig.savefig(file_name, bbox_inches="tight")
    plt.close()


def distance_vs_error(
    root_dir: str, dir_prefix: str, test_set: Tuple, relative: bool = False, remove_outliers: bool = True
):
    metrics = {
        "single_traj_error": [],
        "data_driven_traj_error": [],
        "physics_based_traj_error": [],
        "distance_from_camera": [],
        "angle_from_camera": [],
    }
    cam_space = (
        (0, 2, 3, 4),
        (0, 1, 2, 3, 4, 5),
        (0, 1, 2, 3),
        (0, 1, 2, 3, 4, 5),
        (0, 1, 2, 3, 4, 5),
        (0, 1, 2, 3, 4, 5),
        (0, 1, 2, 3, 4, 5),
        (0, 1, 2, 3, 4, 5),
        (0, 1, 2, 3, 4, 5),
        (0, 1, 2, 3, 4, 5),
    )
    dd_error = []
    pb_error = []
    s_error = []
    for j, test_run in enumerate(test_set):
        for cam_idx in cam_space[j]:
            # Configure trajectory.
            cheetah = test_run[0]
            date = test_run[1]
            trial = test_run[2]
            data_path = f"{date}/{cheetah}/{trial}"
            print(f"{data_path}:")
            # Read the scene data file and the reconstruction data.
            multi_view_data = data_ops.load_pickle(os.path.join(dir_prefix, data_path, "fte_kinematic", "fte.pickle"))
            # Ensure that the camera has been collected.
            if not (
                os.path.exists(os.path.join(dir_prefix, data_path, f"fte_kinematic_{cam_idx}"))
                and os.path.exists(os.path.join(dir_prefix, data_path, f"fte_kinematic_orig_{cam_idx}"))
            ):
                continue
            single_view_data = data_ops.load_pickle(
                os.path.join(dir_prefix, data_path, f"fte_kinematic_orig_{cam_idx}", "fte.pickle")
            )
            data_driven_data = data_ops.load_pickle(
                os.path.join(dir_prefix, data_path, f"fte_kinematic_{cam_idx}", "fte.pickle")
            )
            physics_based_traj_error = []
            for i in range(1):
                physics_based_data = data_ops.load_pickle(
                    os.path.join(
                        dir_prefix, data_path, f"fte_kinetic_{cam_idx}", f"fte{i}.pickle" if i > 0 else "fte.pickle"
                    )
                )
                _, physics_based_traj_error_tmp, _ = misc.traj_error(
                    multi_view_data["positions"].copy(),
                    physics_based_data["positions"].copy(),
                    "physics-based model",
                    centered=relative,
                )
                physics_based_traj_error.append(physics_based_traj_error_tmp)

            physics_based_traj_error = np.mean(physics_based_traj_error, axis=0)
            # physics_based_traj_error_std = np.std(physics_based_traj_error, axis=0)
            # Calculate the trajectory error per bodypart.
            _, single_traj_error, _ = misc.traj_error(
                multi_view_data["positions"].copy(), single_view_data["positions"].copy(), centered=relative
            )
            _, data_driven_traj_error, _ = misc.traj_error(
                multi_view_data["positions"].copy(),
                data_driven_data["positions"].copy(),
                "data-driven model",
                centered=relative,
            )
            dist_from_cam, angle_from_cam = distance_from_camera(
                os.path.join(root_dir, data_path), np.array(multi_view_data["com_pos"]), cam_idx
            )
            metrics["single_traj_error"].append(float(single_traj_error.mean()))
            metrics["data_driven_traj_error"].append(float(data_driven_traj_error.mean()))
            metrics["physics_based_traj_error"].append(float(physics_based_traj_error.mean()))
            metrics["distance_from_camera"].append(np.mean(dist_from_cam))
            metrics["angle_from_camera"].append(np.mean(angle_from_cam))
            dd_error.append(data_driven_traj_error)
            pb_error.append(physics_based_traj_error)
            s_error.append(single_traj_error)

    # Remove the gross outliers that will really skew the data.
    num_points = len(metrics["distance_from_camera"])
    single_outliers = (
        is_outlier(np.array(metrics["single_traj_error"]), 5.0)
        if remove_outliers
        else is_outlier(np.array(metrics["single_traj_error"]), 50.0)
    )
    # single_outliers = np.array(metrics["distance_from_camera"]) > 8.0
    print(f"# Default outliers detected: {np.count_nonzero(single_outliers)}/{num_points}")
    data_driven_outliers = (
        is_outlier(np.array(metrics["data_driven_traj_error"]), 5.0)
        if remove_outliers
        else is_outlier(np.array(metrics["data_driven_traj_error"]), 50.0)
    )
    # data_driven_outliers = single_outliers
    print(f"# Data-driven outliers detected: {np.count_nonzero(data_driven_outliers)}/{num_points}")
    physics_based_outliers = (
        is_outlier(np.array(metrics["physics_based_traj_error"]), 5.0)
        if remove_outliers
        else is_outlier(np.array(metrics["physics_based_traj_error"]), 50.0)
    )
    # physics_based_outliers = single_outliers
    print(f"# Physics-based outliers detected: {np.count_nonzero(physics_based_outliers)}/{num_points}")
    single_dist = np.array(metrics["distance_from_camera"])[~single_outliers]
    single_error = np.array(metrics["single_traj_error"])[~single_outliers]
    fig = plt.figure(figsize=(16, 12), dpi=120)
    plt.scatter(single_dist, single_error, color=misc.plot_color["charcoal"], label="Default")
    plt.plot(
        np.unique(single_dist),
        np.poly1d(np.polyfit(single_dist, single_error, 1))(np.unique(single_dist)),
        color=misc.plot_color["charcoal"],
        linewidth=4.0,
    )
    data_driven_dist = np.array(metrics["distance_from_camera"])[~data_driven_outliers]
    data_driven_error = np.array(metrics["data_driven_traj_error"])[~data_driven_outliers]
    plt.scatter(data_driven_dist, data_driven_error, color=misc.plot_color["green"], label="Data-driven")
    plt.plot(
        np.unique(data_driven_dist),
        np.poly1d(np.polyfit(data_driven_dist, data_driven_error, 1))(np.unique(data_driven_dist)),
        color=misc.plot_color["green"],
        linewidth=4.0,
    )
    physics_based_dist = np.array(metrics["distance_from_camera"])[~physics_based_outliers]
    physics_based_error = np.array(metrics["physics_based_traj_error"])[~physics_based_outliers]
    plt.scatter(physics_based_dist, physics_based_error, color=misc.plot_color["orange"], label="Physics-based")
    plt.plot(
        np.unique(physics_based_dist),
        np.poly1d(np.polyfit(physics_based_dist, physics_based_error, 1))(np.unique(physics_based_dist)),
        color=misc.plot_color["orange"],
        linewidth=4.0,
    )
    plt.legend()
    plt.xlabel("Distance (m)")
    if relative:
        plt.ylabel("MPJPE (mm)")
    else:
        plt.ylabel("MPE (mm)")
    fig.savefig(os.path.join(dir_prefix, "dist_vs_error.pdf"), bbox_inches="tight")
    plt.close()
    default_corr = np.corrcoef(single_dist, single_error)
    data_driven_corr = np.corrcoef(data_driven_dist, data_driven_error)
    physics_based_corr = np.corrcoef(physics_based_dist, physics_based_error)
    print(f"Default correlation coefficient: {default_corr[0, 1]:.3f}")
    print(f"Data-driven correlation coefficient: {data_driven_corr[0, 1]:.3f}")
    print(f"Physics-based correlation coefficient: {physics_based_corr[0, 1]:.3f}")
    print(f"Default error: {np.mean(single_error):.2f} ± {np.std(single_error):.2f}")
    print(f"Data-driven error: {np.mean(data_driven_error):.2f} ± {np.std(data_driven_error):.2f}")
    print(f"Physics-based error: {np.mean(physics_based_error):.2f} ± {np.std(physics_based_error):.2f}")
    align_error_and_plot(s_error, dd_error, pb_error, os.path.join(dir_prefix, "error_distribution.pdf"))


def dataset_post_process(
    root_dir: str,
    dir_prefix: str,
    test_set: Tuple,
    cam_overrides: Optional[List[int]] = None,
    use_best_of: bool = True,
):
    metrics = {
        "single_traj_error": [],
        "data_driven_traj_error": [],
        "physics_based_traj_error": [],
        "single_mpe_mm": [],
        "data_driven_mpe_mm": [],
        "physics_based_mpe_mm": [],
        "physics_based_mpe_mm_std": [],
        "single_mpjpe_mm": [],
        "data_driven_mpjpe_mm": [],
        "physics_based_mpjpe_mm": [],
        "physics_based_mpjpe_mm_std": [],
        "distance_from_camera": [],
        "angle_from_camera": [],
    }
    data_driven_error = []
    physics_based_error = []
    results = {}
    for idx, test_run in enumerate(test_set):
        # Configure trajectory.
        cheetah = test_run[0]
        date = test_run[1]
        trial = test_run[2]
        data_path = f"{date}/{cheetah}/{trial}"
        print(f"{data_path}:")
        with open(os.path.join(root_dir, data_path, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
        cam_idx = metadata["monocular_cam"] if cam_overrides is None else cam_overrides[idx]
        # Read the scene data file and the reconstruction data.
        multi_view_data = data_ops.load_pickle(os.path.join(dir_prefix, data_path, "fte_kinematic", "fte.pickle"))
        single_view_data = data_ops.load_pickle(
            os.path.join(dir_prefix, data_path, f"fte_kinematic_orig_{cam_idx}", "fte.pickle")
        )
        data_driven_data = data_ops.load_pickle(
            os.path.join(dir_prefix, data_path, f"fte_kinematic_{cam_idx}", "fte.pickle")
        )

        # Calculate the trajectory error per bodypart.
        single_mpjpe_mm, _, _ = misc.traj_error(
            multi_view_data["positions"].copy(), single_view_data["positions"].copy(), centered=True
        )
        single_mpe_mm, single_traj_error, single_smooth_error = misc.traj_error(
            multi_view_data["positions"].copy(), single_view_data["positions"].copy()
        )
        data_driven_mpjpe_mm, _, _ = misc.traj_error(
            multi_view_data["positions"].copy(),
            data_driven_data["positions"].copy(),
            "data-driven model",
            centered=True,
        )
        data_driven_mpe_mm, data_driven_traj_error, data_driven_smooth_error = misc.traj_error(
            multi_view_data["positions"].copy(), data_driven_data["positions"].copy(), "data-driven model"
        )
        dist_from_cam, angle_from_cam = distance_from_camera(
            os.path.join(root_dir, data_path), np.array(multi_view_data["com_pos"]), cam_idx
        )
        com_vel_gt = np.array(multi_view_data["com_vel"])
        mpjpe_list = []
        mpe_list = []
        physics_based_traj_error = []
        com_vel_physics_based = []
        physics_based_time = []
        physics_based_smooth_error = []
        physics_based_cost_value = []
        for i in range(1):
            physics_based_data = data_ops.load_pickle(
                os.path.join(
                    dir_prefix, data_path, f"fte_kinetic_{cam_idx}", f"fte{i}.pickle" if i > 0 else "fte.pickle"
                )
            )
            physics_based_mpjpe_mm, _, _ = misc.traj_error(
                multi_view_data["positions"].copy(),
                physics_based_data["positions"].copy(),
                "physics-based model",
                centered=True,
            )
            physics_based_mpe_mm, physics_based_traj_error_tmp, physics_based_smooth = misc.traj_error(
                multi_view_data["positions"].copy(), physics_based_data["positions"].copy(), "physics-based model"
            )
            com_vel_physics_based.append(misc.rmse(com_vel_gt, np.array(physics_based_data["com_vel"])))
            physics_based_traj_error.append(physics_based_traj_error_tmp)
            mpjpe_list.append(physics_based_mpjpe_mm)
            mpe_list.append(physics_based_mpe_mm)
            physics_based_time.append(physics_based_data["processing_time_s"])
            physics_based_smooth_error.append(physics_based_smooth)
            physics_based_cost_value.append(physics_based_data["obj_cost"])
        if use_best_of:
            # Get lowest objective cost.
            idx = np.argmin(physics_based_cost_value)
            # check = np.argmin(np.mean(physics_based_traj_error, axis=1))
            # if check != idx:
            # print(f"Lowest MPE is {check} but lowest cost is {idx}")
            physics_based_mpjpe_mm_std = physics_based_mpjpe_mm.groupby(lambda x: x, axis=1).std()
            physics_based_mpe_mm_std = physics_based_mpe_mm.groupby(lambda x: x, axis=1).std()
            physics_based_mpjpe_mm = mpjpe_list[idx]
            physics_based_mpe_mm = mpe_list[idx]
            physics_based_traj_error = physics_based_traj_error[idx]
            physics_based_sim = com_vel_physics_based[idx]
            physics_based_sim_std = 0.0
            physics_based_time = physics_based_time[idx]
            physics_based_time_std = 0.0
            physics_based_smooth_error = physics_based_smooth_error[idx]
            physics_based_smooth_error_std = 0.0
            metrics["physics_based_mpe_mm_std"].append(physics_based_mpe_mm_std)
            metrics["physics_based_mpjpe_mm_std"].append(physics_based_mpjpe_mm_std)
        else:
            physics_based_mpjpe_mm = pd.concat(mpjpe_list, axis=1)
            physics_based_mpe_mm = pd.concat(mpe_list, axis=1)
            physics_based_mpjpe_mm_std = physics_based_mpjpe_mm.groupby(lambda x: x, axis=1).std()
            physics_based_mpjpe_mm = physics_based_mpjpe_mm.groupby(lambda x: x, axis=1).mean()
            # physics_based_mpjpe_mm = pd.concat([df_tmp1, df_tmp2], axis=1)
            physics_based_mpe_mm_std = physics_based_mpe_mm.groupby(lambda x: x, axis=1).std()
            physics_based_mpe_mm = physics_based_mpe_mm.groupby(lambda x: x, axis=1).mean()
            # physics_based_mpe_mm = pd.concat([df_tmp1, df_tmp2], axis=1)
            physics_based_traj_error = np.mean(physics_based_traj_error, axis=0)
            physics_based_traj_error_std = np.std(physics_based_traj_error, axis=0)
            physics_based_sim = np.mean(com_vel_physics_based, axis=0)
            physics_based_sim_std = np.std(com_vel_physics_based, axis=0)
            physics_based_time_std = np.std(physics_based_time)
            physics_based_time = np.mean(physics_based_time)
            physics_based_smooth_error_std = np.std(physics_based_smooth_error)
            physics_based_smooth_error = np.mean(physics_based_smooth_error)
            metrics["physics_based_mpe_mm_std"].append(physics_based_mpe_mm_std.transpose())
            metrics["physics_based_mpjpe_mm_std"].append(physics_based_mpjpe_mm_std.transpose())
            # physics_based_mpe_mm = pd.concat([df_tmp1, df_tmp2], axis=1)
        metrics["single_mpe_mm"].append(single_mpe_mm.transpose())
        metrics["data_driven_mpe_mm"].append(data_driven_mpe_mm.transpose())
        metrics["physics_based_mpe_mm"].append(physics_based_mpe_mm.transpose())
        metrics["single_mpjpe_mm"].append(single_mpjpe_mm.transpose())
        metrics["data_driven_mpjpe_mm"].append(data_driven_mpjpe_mm.transpose())
        metrics["physics_based_mpjpe_mm"].append(physics_based_mpjpe_mm.transpose())
        metrics["single_traj_error"].append(single_traj_error.mean())
        metrics["data_driven_traj_error"].append(data_driven_traj_error.mean())
        metrics["physics_based_traj_error"].append(physics_based_traj_error.mean())
        metrics["distance_from_camera"].append(dist_from_cam.mean())
        metrics["angle_from_camera"].append(np.mean(angle_from_cam))
        com_vel_single = np.array(single_view_data["com_vel"])
        com_vel_data_driven = np.array(data_driven_data["com_vel"])
        single_sim = misc.rmse(com_vel_gt, com_vel_single)
        data_driven_sim = misc.rmse(com_vel_gt, com_vel_data_driven)
        single_view_time = single_view_data["processing_time_s"]
        data_driven_time = data_driven_data["processing_time_s"]
        data_driven_error.append(data_driven_traj_error)
        physics_based_error.append(physics_based_traj_error)
        print(f"single view CoM cosine similarity: {single_sim}")
        print(f"data-driven model CoM cosine similarity: {data_driven_sim}")
        print(f"physics-based model CoM cosine similarity: {physics_based_sim}")
        results[data_path] = {
            "default": {
                "mpe": round(float(single_mpe_mm.mean()), 1),
                "mpjpe": round(float(single_mpjpe_mm.mean()), 1),
                "CoM vel rmse": round(single_sim, 2),
                "smoothness error": round(single_smooth_error, 1),
                "time": round(single_view_time, 0),
            },
            "data-driven": {
                "mpe": round(float(data_driven_mpe_mm.mean()), 1),
                "mpjpe": round(float(data_driven_mpjpe_mm.mean()), 1),
                "CoM vel rmse": round(data_driven_sim, 2),
                "smoothness error": round(data_driven_smooth_error, 1),
                "time": round(data_driven_time, 0),
            },
            "physics-based": {
                "mpe": f"{round(float(physics_based_mpe_mm.mean()), 1)} \u00B1 {round(float(physics_based_mpe_mm_std.mean()), 1)}",
                "mpjpe": f"{round(float(physics_based_mpjpe_mm.mean()), 1)} \u00B1 {round(float(physics_based_mpjpe_mm_std.mean()), 1)}",
                "CoM vel rmse": f"{round(physics_based_sim, 2)} \u00B1 {round(physics_based_sim_std, 2)}",
                "smoothness error": f"{round(physics_based_smooth_error, 1)} \u00B1 {round(physics_based_smooth_error_std, 1)}",
                "time": f"{round(physics_based_time, 0)} \u00B1 {round(physics_based_time_std, 1)}",
            },
        }

    single_total_error = pd.concat(metrics["single_mpjpe_mm"]).mean()
    model_total_error = pd.concat(metrics["data_driven_mpjpe_mm"]).mean()
    single_total_error_2 = pd.concat(metrics["single_mpe_mm"]).mean()
    model_total_error_2 = pd.concat(metrics["data_driven_mpe_mm"]).mean()
    model_total_error_3 = pd.concat(metrics["physics_based_mpjpe_mm"]).mean()
    model_total_error_3_std = pd.concat(metrics["physics_based_mpjpe_mm_std"]).mean()
    model_total_error_4 = pd.concat(metrics["physics_based_mpe_mm"]).mean()
    model_total_error_4_std = pd.concat(metrics["physics_based_mpe_mm_std"]).mean()
    df = pd.concat([single_total_error, model_total_error], axis=1)
    df.columns = ["Single view", "Data-driven"]
    df2 = pd.concat([single_total_error_2, model_total_error_2], axis=1)
    df2.columns = ["Single view", "Data-driven"]
    df3 = pd.concat([single_total_error, model_total_error_3], axis=1)
    df3.columns = ["Single view", "Physics-based"]
    df4 = pd.concat([single_total_error_2, model_total_error_4], axis=1)
    df4.columns = ["Single view", "Physics-based"]
    dict_of_df = {k: pd.DataFrame(v) for k, v in results.items()}
    results_df = pd.concat(dict_of_df, axis=1)
    results_df.to_csv(os.path.join(dir_prefix, "dataset_results.csv"))
    print(results_df.T)
    fig = plt.figure(figsize=(16, 12), dpi=120)
    ax = df.plot(kind="barh", color=[misc.plot_color["charcoal"], misc.plot_color["orange"]])
    fig = ax.get_figure()
    plt.title(f"Single View ({single_total_error.mean():.1f}) vs Data-driven ({model_total_error.mean():.1f})")
    plt.xlabel("Error (mm)")
    plt.ylabel("Joint")
    fig.savefig(os.path.join(dir_prefix, "data_driven_mpjpe_result.pdf"), bbox_inches="tight")
    plt.close()

    fig = plt.figure(figsize=(16, 12), dpi=120)
    ax = df2.plot(kind="barh", color=[misc.plot_color["charcoal"], misc.plot_color["orange"]])
    fig = ax.get_figure()
    plt.title(f"Single View ({single_total_error_2.mean():.1f}) vs Data-driven ({model_total_error_2.mean():.1f})")
    plt.xlabel("Error (mm)")
    plt.ylabel("Joint")
    fig.savefig(os.path.join(dir_prefix, "data_driven_mpe_result.pdf"), bbox_inches="tight")
    plt.close()

    fig = plt.figure(figsize=(16, 12), dpi=120)
    ax = df3.plot(kind="barh", color=[misc.plot_color["charcoal"], misc.plot_color["orange"]])
    fig = ax.get_figure()
    plt.title(
        f"Single View ({single_total_error.mean():.1f}) vs Physics-based ({model_total_error_3.mean():.1f} \u00B1 {model_total_error_3_std.mean():.1f})"
    )
    plt.xlabel("Error (mm)")
    plt.ylabel("Joint")
    fig.savefig(os.path.join(dir_prefix, "physics_based_mpjpe_result.pdf"), bbox_inches="tight")
    plt.close()

    fig = plt.figure(figsize=(16, 12), dpi=120)
    ax = df4.plot(kind="barh", color=[misc.plot_color["charcoal"], misc.plot_color["orange"]])
    fig = ax.get_figure()
    plt.title(
        f"Single View ({single_total_error_2.mean():.1f}) vs Physics-based ({model_total_error_4.mean():.1f} \u00B1 {model_total_error_4_std.mean():.1f})"
    )
    plt.xlabel("Error (mm)")
    plt.ylabel("Joint")
    fig.savefig(os.path.join(dir_prefix, "physics_based_mpe_result.pdf"), bbox_inches="tight")
    plt.close()

    fig = plt.figure(figsize=(16, 12), dpi=120)
    plt.scatter(metrics["distance_from_camera"], metrics["single_traj_error"], color=misc.plot_color["charcoal"])
    plt.plot(
        np.unique(metrics["distance_from_camera"]),
        np.poly1d(np.polyfit(metrics["distance_from_camera"], metrics["single_traj_error"], 1))(
            np.unique(metrics["distance_from_camera"])
        ),
        color=misc.plot_color["charcoal"],
    )
    plt.scatter(metrics["distance_from_camera"], metrics["data_driven_traj_error"], color=misc.plot_color["green"])
    plt.plot(
        np.unique(metrics["distance_from_camera"]),
        np.poly1d(np.polyfit(metrics["distance_from_camera"], metrics["data_driven_traj_error"], 1))(
            np.unique(metrics["distance_from_camera"])
        ),
        color=misc.plot_color["green"],
    )
    plt.scatter(metrics["distance_from_camera"], metrics["physics_based_traj_error"], color=misc.plot_color["red"])
    plt.plot(
        np.unique(metrics["distance_from_camera"]),
        np.poly1d(np.polyfit(metrics["distance_from_camera"], metrics["physics_based_traj_error"], 1))(
            np.unique(metrics["distance_from_camera"])
        ),
        color=misc.plot_color["red"],
    )
    fig.savefig(os.path.join(dir_prefix, "dist_vs_error.pdf"), bbox_inches="tight")
    plt.close()

    align_error_and_plot(data_driven_error, os.path.join(dir_prefix, "data_driven_error_traj.pdf"))
    align_error_and_plot(physics_based_error, os.path.join(dir_prefix, "physics_based_error_traj.pdf"))


def run_data_driven_ablation_study(root_dir: str, dir_prefix: str, test_set: Tuple):
    results = {
        "mpjpe": [],
        "mpjpe_std": [],
        "mpe": [],
        "mpe_std": [],
        "centroid_vel_rmse": [],
        "centroid_vel_rmse_std": [],
        "smoothness_error": [],
        "smoothness_error_std": [],
        "optimisation_time": [],
        "optimisation_time_std": [],
    }
    time0 = time()
    test_cases = ((True, True), (True, False), (False, True), (False, False))
    for scenario in test_cases:
        test_mpe = []
        test_mpjpe = []
        test_time = []
        test_smoothness = []
        test_centroid_vel_rmse = []
        for test_run in test_set:
            # Force garbage collection so that the repeated model creation does not overflow the memory!
            gc.collect()
            # Configure trajectory.
            cheetah = test_run[0]
            date = test_run[1]
            trial = test_run[2]
            data_path = f"{date}/{cheetah}/{trial}"
            estimator = opt.init_trajectory(
                root_dir,
                data_path,
                cheetah,
                solver_path="C:/Users/DSLZI/Downloads/idaes-solvers-windows-x86_64/ipopt.exe",
                monocular_enable=True,
                include_camera_constraints=True,
                kinetic_dataset=False,
                kinematic_model=True,
            )
            opt.estimate_kinematics(
                estimator,
                monocular_constraints=True,
                solver_output=False,
                disable_pose_prior=scenario[0],
                disable_motion_prior=scenario[1],
                out_dir_prefix=dir_prefix,
            )
            # Get results.
            multi_view_data = data_ops.load_pickle(os.path.join(dir_prefix, data_path, "fte_kinematic", "fte.pickle"))
            data_driven_data = data_ops.load_pickle(
                os.path.join(dir_prefix, data_path, f"fte_kinematic_{estimator.scene.cam_idx}", "fte.pickle")
            )
            data_driven_mpjpe_mm, _, _ = misc.traj_error(
                multi_view_data["positions"].copy(),
                data_driven_data["positions"].copy(),
                "data-driven model",
                centered=True,
            )
            data_driven_mpe_mm, _, smoothness = misc.traj_error(
                multi_view_data["positions"].copy(), data_driven_data["positions"].copy(), "data-driven model"
            )
            com_vel_gt = np.array(multi_view_data["com_vel"])
            com_vel_data_driven = np.array(data_driven_data["com_vel"])
            data_driven_sim = misc.rmse(com_vel_gt, com_vel_data_driven)
            test_time.append(data_driven_data["processing_time_s"])
            test_mpe.append(float(data_driven_mpe_mm.mean()))
            test_mpjpe.append(float(data_driven_mpjpe_mm.mean()))
            test_smoothness.append(smoothness)
            test_centroid_vel_rmse.append(data_driven_sim)
        results["optimisation_time"].append(np.mean(test_time))
        results["optimisation_time_std"].append(np.std(test_time))
        results["mpe"].append(np.mean(test_mpe))
        results["mpe_std"].append(np.std(test_mpe))
        results["mpjpe"].append(np.mean(test_mpjpe))
        results["mpjpe_std"].append(np.std(test_mpjpe))
        results["centroid_vel_rmse"].append(np.mean(test_centroid_vel_rmse))
        results["centroid_vel_rmse_std"].append(np.std(test_centroid_vel_rmse))
        results["smoothness_error"].append(np.mean(test_smoothness))
        results["smoothness_error_std"].append(np.std(test_smoothness))
    data_ops.save_pickle(os.path.join(dir_prefix, "data_driven_ablation_study.pickle"), results)
    print(results)
    time1 = time()
    print(f"Run through all videos took {time1 - time0:.2f}s")


def run_physics_based_ablation_study(root_dir: str, dir_prefix: str, test_set: Tuple):
    results = {
        "mpjpe": [],
        "mpjpe_std": [],
        "mpe": [],
        "mpe_std": [],
        "centroid_vel_rmse": [],
        "centroid_vel_rmse_std": [],
        "smoothness_error": [],
        "smoothness_error_std": [],
        "optimisation_time": [],
        "optimisation_time_std": [],
    }
    time0 = time()
    test_cases = ((True, True), (True, False), (False, True), (False, False))
    for scenario in test_cases:
        test_mpe = []
        test_mpjpe = []
        test_time = []
        test_smoothness = []
        test_centroid_vel_rmse = []
        for test_run in test_set:
            # Force garbage collection so that the repeated model creation does not overflow the memory!
            gc.collect()
            # Configure trajectory.
            cheetah = test_run[0]
            date = test_run[1]
            trial = test_run[2]
            data_path = f"{date}/{cheetah}/{trial}"
            with open(os.path.join(root_dir, data_path, "metadata.json"), "r", encoding="utf-8") as f:
                metadata = json.load(f)
            cam_idx = metadata["monocular_cam"]
            estimator = opt.init_trajectory(
                root_dir,
                data_path,
                cheetah,
                solver_path="C:/Users/DSLZI/Downloads/idaes-solvers-windows-x86_64/ipopt.exe",
                monocular_enable=True,
                include_camera_constraints=True,
                enable_eom_slack=True,
                bound_eom_error=(-2.0, 2.0),
                kinetic_dataset=False,
                kinematic_model=False,
            )
            opt.estimate_kinetics(
                estimator,
                solver_output=False,
                init_torques=False,
                init_prev_kinematic_solution=False,
                joint_estimation=True,
                auto=False,
                disable_pose_prior=scenario[0],
                disable_motion_prior=scenario[1],
                out_dir_prefix=dir_prefix,
            )
            # Get results.
            multi_view_data = data_ops.load_pickle(os.path.join(dir_prefix, data_path, "fte_kinematic", "fte.pickle"))
            physics_based_data = data_ops.load_pickle(
                os.path.join(dir_prefix, data_path, f"fte_kinetic_{cam_idx}", "fte.pickle")
            )
            physics_based_mpjpe_mm, _, _ = misc.traj_error(
                multi_view_data["positions"].copy(),
                physics_based_data["positions"].copy(),
                "physics-based model",
                centered=True,
            )
            physics_based_mpe_mm, _, smoothness = misc.traj_error(
                multi_view_data["positions"].copy(), physics_based_data["positions"].copy(), "physics-based model"
            )
            com_vel_gt = np.array(multi_view_data["com_vel"])
            com_vel_data_driven = np.array(physics_based_data["com_vel"])
            data_driven_sim = misc.rmse(com_vel_gt, com_vel_data_driven)
            test_time.append(physics_based_data["processing_time_s"])
            test_mpe.append(float(physics_based_mpe_mm.mean()))
            test_mpjpe.append(float(physics_based_mpjpe_mm.mean()))
            test_smoothness.append(smoothness)
            test_centroid_vel_rmse.append(data_driven_sim)
        results["optimisation_time"].append(np.mean(test_time))
        results["optimisation_time_std"].append(np.std(test_time))
        results["mpe"].append(np.mean(test_mpe))
        results["mpe_std"].append(np.std(test_mpe))
        results["mpjpe"].append(np.mean(test_mpjpe))
        results["mpjpe_std"].append(np.std(test_mpjpe))
        results["centroid_vel_rmse"].append(np.mean(test_centroid_vel_rmse))
        results["centroid_vel_rmse_std"].append(np.std(test_centroid_vel_rmse))
        results["smoothness_error"].append(np.mean(test_smoothness))
        results["smoothness_error_std"].append(np.std(test_smoothness))
    data_ops.save_pickle(os.path.join(dir_prefix, "physics_based_ablation_study.pickle"), results)
    print(results)
    time1 = time()
    print(f"Run through all videos took {time1 - time0:.2f}s")


def run_grid_search(root_dir: str, dir_prefix: str, test_set: Tuple):
    n_comps = (1, 2, 3, 4, 5, 6, 7)
    window_size = (1, 2, 3, 4, 5, 6, 7)
    lasso = (True, False)
    results = {
        "gmm_train_likelihood": [],
        "gmm_validation_likelihood": [],
        "lr_train_rmse": [],
        "lr_validation_rmse": [],
        "lr_non_zeros": [],
        "mpjpe": [],
        "mpe": [],
        "optimisation_time": [],
    }
    time0 = time()
    for num_comp in n_comps:
        current_gmm_train_likelihood = 0
        current_gmm_validation_likelihood = 0
        for sparse_sol in lasso:
            current_lr_non_zeros = 0
            current_lr_train_rmse = 0
            current_lr_validation_rmse = 0
            for w_size in window_size:
                print(f"Evaluate dataset with the following hyper-parameters: {num_comp, w_size, sparse_sol}")
                if w_size == 0 and sparse_sol is False:
                    # No point running this again because the motion model has been turned off for window size 0.
                    continue
                avg_mpe = []
                avg_mpjpe = []
                avg_time = []
                for test_run in test_set:
                    # Force garbage collection so that the repeated model creation does not overflow the memory!
                    gc.collect()
                    # Configure trajectory.
                    cheetah = test_run[0]
                    date = test_run[1]
                    trial = test_run[2]
                    data_path = f"{date}/{cheetah}/{trial}"
                    estimator = opt.init_trajectory(
                        root_dir,
                        data_path,
                        cheetah,
                        solver_path="C:/Users/DSLZI/Downloads/idaes-solvers-windows-x86_64/ipopt.exe",
                        monocular_enable=True,
                        include_camera_constraints=True,
                        kinetic_dataset=False,
                        kinematic_model=True,
                    )
                    opt.estimate_kinematics(
                        estimator,
                        monocular_constraints=True,
                        solver_output=False,
                        disable_pose_prior=True if num_comp == 0 else False,
                        disable_motion_prior=True if w_size == 0 else False,
                        pose_model_num_components=num_comp,
                        motion_model_window_size=w_size,
                        motion_model_sparse_solution=sparse_sol,
                        out_dir_prefix=dir_prefix,
                    )
                    # Get results.
                    multi_view_data = data_ops.load_pickle(
                        os.path.join(dir_prefix, data_path, "fte_kinematic", "fte.pickle")
                    )
                    data_driven_data = data_ops.load_pickle(
                        os.path.join(dir_prefix, data_path, f"fte_kinematic_{estimator.scene.cam_idx}", "fte.pickle")
                    )
                    data_driven_mpjpe_mm, _, _ = misc.traj_error(
                        multi_view_data["positions"].copy(),
                        data_driven_data["positions"].copy(),
                        "data-driven model",
                        centered=True,
                    )
                    data_driven_mpe_mm, _, _ = misc.traj_error(
                        multi_view_data["positions"].copy(), data_driven_data["positions"].copy(), "data-driven model"
                    )
                    avg_time.append(data_driven_data["processing_time_s"])
                    avg_mpe.append(float(data_driven_mpe_mm.mean()))
                    avg_mpjpe.append(float(data_driven_mpjpe_mm.mean()))
                    current_lr_non_zeros = estimator.motion_model.model_non_zeros if estimator.motion_model else None
                    current_lr_train_rmse = estimator.motion_model.train_rmse if estimator.motion_model else None
                    current_lr_validation_rmse = (
                        estimator.motion_model.validation_rmse if estimator.motion_model else None
                    )
                    current_gmm_train_likelihood = (
                        estimator.pose_model.log_likelihood_train_set if estimator.pose_model else None
                    )
                    current_gmm_validation_likelihood = (
                        estimator.pose_model.log_likelihood_validation_set if estimator.pose_model else None
                    )
                results["optimisation_time"].append(sum(avg_time) / len(test_set))
                results["mpe"].append(sum(avg_mpe) / len(test_set))
                results["mpjpe"].append(sum(avg_mpjpe) / len(test_set))
                if num_comp == n_comps[0]:
                    # Only add LR results for the first run through the data. No need to duplicate results.
                    results["lr_non_zeros"].append(current_lr_non_zeros)
                    results["lr_train_rmse"].append(current_lr_train_rmse)
                    results["lr_validation_rmse"].append(current_lr_validation_rmse)
        results["gmm_train_likelihood"].append(current_gmm_train_likelihood)
        results["gmm_validation_likelihood"].append(current_gmm_validation_likelihood)
    data_ops.save_pickle(os.path.join(dir_prefix, "grid_search.pickle"), results)
    print(results)
    time1 = time()
    print(f"Run through all videos took {time1 - time0:.2f}s")


def run_monocular_all(root_dir: str, dir_prefix: str, test_set: Tuple):
    time0 = time()
    test_types = ("default", "data-driven", "physics-based")
    cam_space = (
        (0, 2, 3, 4),
        (0, 1, 2, 3, 4, 5),
        (0, 1, 2, 3),
        (0, 1, 2, 3, 4, 5),
        (0, 1, 2, 3, 4, 5),
        (0, 1, 2, 3, 4, 5),
        (0, 1, 2, 3, 4, 5),
        (0, 1, 2, 3, 4, 5),
        (0, 1, 2, 3, 4, 5),
        (0, 1, 2, 3, 4, 5),
    )
    auto_contact_detection = True
    for i, test_run in enumerate(test_set):
        # Configure trajectory.
        cheetah = test_run[0]
        date = test_run[1]
        trial = test_run[2]
        data_path = f"{date}/{cheetah}/{trial}"
        # _, _, _, _, _, n_cams, _ = misc.find_scene_file(os.path.join(root_dir, data_path))
        for monocular_cam in cam_space[i]:
            multiplyer = 0
            for t in test_types:
                # Force garbage collection so that the repeated model creation does not overflow the memory!
                gc.collect()
                success = True
                if t == "default":
                    estimator = opt.init_trajectory(
                        root_dir,
                        data_path,
                        cheetah,
                        solver_path="C:/Users/DSLZI/Downloads/idaes-solvers-windows-x86_64/ipopt.exe",
                        monocular_enable=True,
                        include_camera_constraints=True,
                        override_monocular_cam=monocular_cam,
                        kinetic_dataset=False,
                        kinematic_model=True,
                    )
                    success = opt.estimate_kinematics(estimator, out_dir_prefix=dir_prefix)
                elif t == "data-driven":
                    estimator = opt.init_trajectory(
                        root_dir,
                        data_path,
                        cheetah,
                        solver_path="C:/Users/DSLZI/Downloads/idaes-solvers-windows-x86_64/ipopt.exe",
                        monocular_enable=True,
                        include_camera_constraints=True,
                        override_monocular_cam=monocular_cam,
                        kinetic_dataset=False,
                        kinematic_model=True,
                    )
                    success = opt.estimate_kinematics(
                        estimator, monocular_constraints=True, solver_output=False, out_dir_prefix=dir_prefix
                    )
                elif t == "physics-based":
                    while True:
                        estimator = opt.init_trajectory(
                            root_dir,
                            data_path,
                            cheetah,
                            solver_path="C:/Users/DSLZI/Downloads/idaes-solvers-windows-x86_64/ipopt.exe",
                            monocular_enable=True,
                            enable_eom_slack=True,
                            bound_eom_error=(-2.0, 2.0),
                            include_camera_constraints=True,
                            override_monocular_cam=monocular_cam,
                            kinetic_dataset=False,
                            kinematic_model=False,
                        )
                        try:
                            if auto_contact_detection:
                                opt.determine_contacts(
                                    estimator, monocular=True, verbose=True, out_dir_prefix=dir_prefix
                                )
                            success = opt.estimate_kinetics(
                                estimator,
                                init_torques=False,
                                init_prev_kinematic_solution=False,
                                out_dir_prefix=dir_prefix,
                                solver_output=False,
                                auto=auto_contact_detection,
                                out_fname=f"fte{multiplyer}" if multiplyer > 0 else "fte",
                                joint_estimation=True,
                            )
                        # except ValueError:
                        #     continue
                        except FileNotFoundError:
                            success = False
                            break
                        break
                    multiplyer += 1
                if not success:
                    break
    time1 = time()
    print(f"Run through all videos took {time1 - time0:.2f}s")


def kinetic_analysis(root_dir: str, dir_prefix: str):
    import results

    kinetic_set = (
        ("arabia", "2009_09_07", "06"),
        ("shiraz", "2009_09_07", "04"),
        ("shiraz", "2009_09_08", "04"),
        ("shiraz", "2009_09_11", "01"),
        ("shiraz", "2009_09_11", "02"),
    )

    def load_cheetah(cheetah: str, data_path: str, dir_prefix: str):
        estimator = opt.init_trajectory(
            root_dir,
            data_path,
            cheetah,
            solver_path="C:/Users/DSLZI/Downloads/idaes-solvers-windows-x86_64/ipopt.exe",
            include_camera_constraints=False,
            kinetic_dataset=True,
            kinematic_model=False,
            bound_eom_error=(-2.0, 2.0),
        )
        data_dir = (
            estimator.params.data_dir
            if dir_prefix is None
            else os.path.join(dir_prefix, estimator.params.data_dir.split("cheetah_videos")[1][1:])
        )
        estimator.load(os.path.join(data_dir, "fte_kinetic"))
        return estimator, results.contact_json_conversion(
            estimator.model, os.path.join(root_dir, estimator.data_path, "metadata.json")
        )

    gait_array = []
    for test_run in kinetic_set:
        # Configure trajectory.
        cheetah = test_run[0]
        date = test_run[1]
        trial = test_run[2]
        data_path = f"kinetic_dataset/{date}/{cheetah}/trial{trial}"
        estimator, cheetah_contact = load_cheetah(cheetah, data_path, dir_prefix)
        gait_array.append(results.gait_analysis(estimator.model, cheetah_contact))
        # Get 2D metrics based on hand labelled data.
        try:
            mean_error, med_error = results.metrics(
                root_dir,
                data_path,
                estimator.params.start_frame,
                estimator.params.end_frame,
                estimator.params.dlc_thresh,
                type_3D_gt="fte_kinetic",
                out_dir_prefix=dir_prefix,
            )
            print(f"{data_path} 2D metrics: ({mean_error:.3f}, {med_error:.3f})")
        except AssertionError:
            print(f"No hand labelled data for test: {data_path}")
        # Display kinematic error.
        results.kinematic_error(estimator.model, dir_prefix, data_path)
        # Plot torques.
        results.plot_torques(estimator.model, os.path.join(dir_prefix, data_path))
        # Load estimated GRF.
        data_dir = (
            estimator.params.data_dir
            if dir_prefix is None
            else os.path.join(dir_prefix, estimator.params.data_dir.split("/cheetah_videos/")[1])
        )
        estimator.load(os.path.join(data_dir, "fte_grf"))
        # Plot comparison of GRF.
        results.grf_error(
            estimator.model, os.path.join(root_dir, estimator.data_path), os.path.join(dir_prefix, data_path)
        )
    # Plot gait analysis for comparison with biomechanic research.
    results.plot_gait_attributes(gait_array, dir_prefix=dir_prefix)


def run_kinetic(root_dir: str, dir_prefix: str):
    kinetic_set = (
        ("arabia", "2009_09_07", "06"),
        ("shiraz", "2009_09_07", "04"),
        ("shiraz", "2009_09_08", "04"),
        ("shiraz", "2009_09_11", "01"),
        ("shiraz", "2009_09_11", "02"),
    )
    time0 = time()
    for test_run in kinetic_set:
        # Force garbage collection so that the repeated model creation does not overflow the memory!
        gc.collect()
        # Configure trajectory.
        cheetah = test_run[0]
        date = test_run[1]
        trial = test_run[2]
        data_path = f"kinetic_dataset/{date}/{cheetah}/trial{trial}"
        estimator = opt.init_trajectory(
            root_dir, data_path, cheetah, solver_path="C:/Users/DSLZI/Downloads/idaes-solvers-windows-x86_64/ipopt.exe", kinetic_dataset=True, kinematic_model=True, bound_eom_error=(-2.0, 2.0)
        )
        ret = opt.estimate_kinematics(estimator, solver_output=False, out_dir_prefix=dir_prefix)
        if ret:
            estimator = opt.init_trajectory(
                root_dir, data_path, cheetah, solver_path="C:/Users/DSLZI/Downloads/idaes-solvers-windows-x86_64/ipopt.exe", kinetic_dataset=True, kinematic_model=False, bound_eom_error=(-2.0, 2.0)
            )
            ret = opt.estimate_kinetics(
                estimator,
                init_torques=False,
                init_prev_kinematic_solution=True,
                out_dir_prefix=dir_prefix,
                solver_output=False,
                ground_constraint=True,
                synthesised_grf=True,
                auto=False,
                joint_estimation=False,
            )
            if ret:
                estimator = opt.init_trajectory(
                    root_dir,
                    data_path,
                    cheetah,
                    solver_path="C:/Users/DSLZI/Downloads/idaes-solvers-windows-x86_64/ipopt.exe",
                    kinetic_dataset=True,
                    kinematic_model=False,
                    bound_eom_error=(-0.1, 0.1),
                )
                ret = opt.estimate_grf(estimator, solver_output=False, out_dir_prefix=dir_prefix)
    time1 = time()
    print(f"Run through all videos took {time1 - time0:.2f}s")


def run_monocular(root_dir: str, dir_prefix: str, test_set: Tuple, cam_overrides: Optional[List[int]] = None):
    time0 = time()
    for idx, test_run in enumerate(test_set):
        # Configure trajectory.
        cheetah = test_run[0]
        date = test_run[1]
        trial = test_run[2]
        data_path = f"{date}/{cheetah}/{trial}"
        test_types = ("ground-truth", "default", "data-driven", "physics-based")
        multiplyer = 0
        monocular_cam = None
        if cam_overrides is not None:
            monocular_cam = cam_overrides[idx]
        for t in test_types:
            # Force garbage collection so that the repeated model creation does not overflow the memory!
            gc.collect()
            if t == "ground-truth":
                estimator = opt.init_trajectory(
                    root_dir,
                    data_path,
                    cheetah,
                    solver_path="C:/Users/DSLZI/Downloads/idaes-solvers-windows-x86_64/ipopt.exe",
                    include_camera_constraints=True,
                    kinetic_dataset=False,
                    kinematic_model=True,
                )
                opt.estimate_kinematics(estimator, out_dir_prefix=dir_prefix, solver_output=False)
            elif t == "default":
                estimator = opt.init_trajectory(
                    root_dir,
                    data_path,
                    cheetah,
                    solver_path="C:/Users/DSLZI/Downloads/idaes-solvers-windows-x86_64/ipopt.exe",
                    monocular_enable=True,
                    include_camera_constraints=True,
                    override_monocular_cam=monocular_cam,
                    kinetic_dataset=False,
                    kinematic_model=True,
                )
                opt.estimate_kinematics(estimator, out_dir_prefix=dir_prefix, solver_output=False)
            elif t == "data-driven":
                estimator = opt.init_trajectory(
                    root_dir,
                    data_path,
                    cheetah,
                    solver_path="C:/Users/DSLZI/Downloads/idaes-solvers-windows-x86_64/ipopt.exe",
                    monocular_enable=True,
                    include_camera_constraints=True,
                    override_monocular_cam=monocular_cam,
                    kinetic_dataset=False,
                    kinematic_model=True,
                )
                opt.estimate_kinematics(
                    estimator, monocular_constraints=True, solver_output=False, out_dir_prefix=dir_prefix
                )
            elif t == "physics-based":
                while True:
                    estimator = opt.init_trajectory(
                        root_dir,
                        data_path,
                        cheetah,
                        solver_path="C:/Users/DSLZI/Downloads/idaes-solvers-windows-x86_64/ipopt.exe",
                        monocular_enable=True,
                        enable_eom_slack=True,
                        bound_eom_error=(-2.0, 2.0),
                        include_camera_constraints=True,
                        override_monocular_cam=monocular_cam,
                        kinetic_dataset=False,
                        kinematic_model=False,
                    )
                    opt.determine_contacts(estimator, monocular=True, verbose=False, out_dir_prefix=dir_prefix)
                    try:
                        ret = opt.estimate_kinetics(
                            estimator,
                            init_torques=False,
                            init_prev_kinematic_solution=True,
                            out_dir_prefix=dir_prefix,
                            solver_output=False,
                            auto=True,
                            out_fname=f"fte{multiplyer}" if multiplyer > 0 else "fte",
                            joint_estimation=True,
                        )
                    except ValueError:
                        continue
                    if ret:
                        break
                multiplyer += 1
    time1 = time()
    print(f"Run through all videos took {time1 - time0:.2f}s")


def run_acinoset(root_dir: str, out_dir_prefix: str):
    """
    Runs through the video list in AcinoSet and performs the 3D reconstruction for each video.
    Args:
        root_dir: The root directory where the videos are stored, along with the `pose_3d_functions.pickle` file, and `gt_labels` directory.
        video_data: The list of videos stored in a dictionary with a key `test_dirs` for each directory in AcinoSet.
        out_dir_prefix: Used to change the output directory from the root. This is often used if you have the cheetah data stored in a location and you want the output to be saved elsewhere.
    """
    video_set = data_ops.load_dill(os.path.join(root_dir.split("/cheetah_videos")[0], "test_set.pickle"))
    tests = video_set["test_dirs"]
    manually_selected_frames = {
        "2019_03_03/phantom/run": (100, 220),
        # "2017_12_12/top/cetane/run1_1": (100, 241),
        # "2019_03_05/jules/run": (58, 176),
        "2019_03_09/lily/run": (80, 170),
        # "2017_09_03/top/zorro/run1_2": (20, 120),
        "2017_08_29/top/phantom/run1_1": (20, 160),
        "2017_12_21/top/lily/run1": (10, 105),
        "2017_12_21/bottom/jules/flick2_2": (5, 150),
        # "2019_03_03/menya/run": (20, 130),
        # "2017_12_10/top/phantom/run1": (30, 130),
        "2017_12_10/top/zorro/flick1": (115, 210),
        "2017_12_10/bottom/zorro/flick2": (5, 140),
        "2017_09_03/bottom/zorro/run2_1": (130, 270),
        # "2019_02_27/ebony/run": (20, 120),
        "2017_12_09/bottom/phantom/run2": (20, 115),
        "2017_09_03/bottom/zorro/run2_3": (5, 150),
        "2017_08_29/top/jules/run1_1": (10, 110),
        "2017_09_02/top/jules/run1": (10, 110),
        "2019_03_07/menya/run": (60, 160),
        "2017_09_02/top/phantom/run1_2": (20, 160),
        # "2019_03_05/lily/run": (150, 250),
        # "2017_12_12/top/cetane/run1_2": (3, 202),
        "2019_03_07/phantom/run": (100, 200),
        "2019_02_27/romeo/run": (40, 150),
        "2019_02_27/romeo/flick": (10, 150),
        "2017_08_29/top/jules/run1_2": (30, 130),
        "2017_12_16/top/cetane/run1": (110, 210),
        # "2017_12_16/bottom/cetane/flick2": (10, 150),
        # "2017_09_02/top/phantom/run1_1": (33, 150),
        # "2017_09_02/top/phantom/run1_3": (35, 135),
        # "2017_09_03/top/zorro/run1_1": (10, 150),
        "2019_02_27/kiara/run": (20, 100),
        "2017_09_02/bottom/jules/run2": (50, 160),
        "2017_09_03/bottom/zorro/run2_2": (32, 141),
        # "2019_03_07/menya/flick": (20, 150),
        # "2019_03_05/lily/flick": (100, 200),
        # "2017_08_29/top/zorro/flick1_2": (20, 140),
        # "2017_09_02/bottom/phantom/flick2_1": (5, 100),
        # "2017_12_12/bottom/big_girl/flick2": (30, 100),
        # "2019_03_03/phantom/flick": (250, 350),
        # "2019_03_09/lily/flick": (10, 100),
        "2019_03_09/jules/flick1": (40, 160),
        # "2017_09_03/top/zorro/flick1_1": (62, 150),
        "2017_09_03/bottom/zorro/flick2": (10, 100),
        # "2017_12_21/top/lily/flick1": (40, 180),
        # "2017_12_21/bottom/jules/flick2_1": (50, 200),
        # "2017_09_03/top/phantom/flick1": (50, 150),
        # "2017_09_02/top/jules/flick1_1": (60, 200),
        # "2017_09_02/bottom/jules/flick2_1": (20, 85),
        # "2017_08_29/top/phantom/flick1_1": (50, 200)
        "2017_08_29/bottom/zorro/flick2": (75, 135),
        "2017_12_09/bottom/jules/flick2": (5, 75),
        "2017_12_17/bottom/zorro/flick2": (5, 145),
    }

    bad_videos = ()
    time0 = time()
    print("Run reconstruction on all videos...")
    for test_vid in tqdm(tests):
        # Force garbage collection so that the repeated model creation does not overflow the memory!
        gc.collect()
        current_dir = test_vid.split("/cheetah_videos/")[1]
        # Filter parameters based on past experience.
        if current_dir in bad_videos:
            # Skip these videos because of erroneous input data.
            continue
        if current_dir in set(manually_selected_frames.keys()):
            start = manually_selected_frames[current_dir][0]
            end = manually_selected_frames[current_dir][1]
            estimator = opt.init_trajectory(
                root_dir,
                current_dir,
                "acinoset",
                solver_path="C:/Users/DSLZI/Downloads/idaes-solvers-windows-x86_64/ipopt.exe",
                kinetic_dataset=False,
                start_frame=start,
                end_frame=end,
                shutter_delay_estimation=False,
                enable_ppm=True if "flick" in current_dir else False,
                kinematic_model=True,
            )
            success = opt.estimate_kinematics(estimator, solver_output=False, out_dir_prefix=out_dir_prefix)
            if not success:
                print(f"Failed to find optimal solution for test: {current_dir}")

    time1 = time()
    print(f"Run through all videos took {time1 - time0:.2f}s")


if __name__ == "__main__":
    parser = ArgumentParser(description="Monocular 3D Reconstruction of Cheetahs in the Wild")
    parser.add_argument("--root_dir", type=str, help="The data directory path to all cheetah videos.")
    parser.add_argument("--out_dir_prefix", type=str, help="The directory to place the outputs.")
    parser.add_argument(
        "--override_default_cam",
        action="store_true",
        help="A flag to override the default cameras used for monocular reconstruction.",
    )
    parser.add_argument(
        "--run_acinoset",
        action="store_true",
        help="A flag to run a subset of AcinoSet that forms the dataset used for train/test procedures in monocular 3D reconstruction.",
    )
    parser.add_argument(
        "--run_monocular",
        action="store_true",
        help="A flag to run a subset of AcinoSet that forms the dataset used for train/test procedures in monocular 3D reconstruction.",
    )
    parser.add_argument(
        "--run_kinetic",
        action="store_true",
        help="A flag to run a subset of AcinoSet that forms the dataset used for train/test procedures in monocular 3D reconstruction.",
    )
    parser.add_argument(
        "--run_analysis",
        action="store_true",
        help="A flag to run a subset of AcinoSet that forms the dataset used for train/test procedures in monocular 3D reconstruction.",
    )
    parser.add_argument(
        "--run_grid_search",
        action="store_true",
        help="A flag to run a subset of AcinoSet that forms the dataset used for train/test procedures in monocular 3D reconstruction.",
    )
    parser.add_argument(
        "--run_physics_based_ablation_study",
        action="store_true",
        help="A flag to run a subset of AcinoSet that forms the dataset used for train/test procedures in monocular 3D reconstruction.",
    )
    parser.add_argument(
        "--run_data_driven_ablation_study",
        action="store_true",
        help="A flag to run a subset of AcinoSet that forms the dataset used for train/test procedures in monocular 3D reconstruction.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="A flag that enables the re-generation of the results prior to post-processing for output metrics. Note you will need to have the original AcinoSet data and the optimisation framework setup to do this.",
    )
    args = parser.parse_args()

    dataset = (
        ("jules", "2017_12_09/bottom", "flick2"),
        ("jules", "2019_03_09", "flick1"),
        ("phantom", "2019_03_03", "run"),  # curve run (acceleration)
        ("phantom", "2017_09_02/top", "run1_2"),  # straight run (deceleration)
        ("jules", "2017_08_29/top", "run1_2"),  # straight run (deceleration/trot)
        ("phantom", "2017_08_29/top", "run1_1"),  # straight run (steady-state slow)
        ("jules", "2017_08_29/top", "run1_1"),  # straight run (steady-state fast)
        ("jules", "2017_09_02/top", "run1"),  # straight run (steady-state fast)
        ("phantom", "2019_03_07", "run"),  # straight run (steady-state slow)
        ("jules", "2017_09_02/bottom", "run2"),
    )  # straight run (steady-state medium)
    monocular_cam_overrides = None
    if args.override_default_cam:
        monocular_cam_overrides = [0, 0, 0, 3, 3, 3, 5, 0, 3, 0]
    if args.run_acinoset:
        print("Running AcinoSet subset")
        if args.clean:
            print("Generating the 3D reconstructions...this will take a while!")
            run_acinoset(os.path.normpath(args.root_dir), os.path.normpath(args.out_dir_prefix))
        print(validate_dataset(os.path.normpath(args.out_dir_prefix)))
    if args.run_monocular:
        print("Running monocular procedure")
        if args.clean:
            print("Generating the 3D reconstructions...this will take a while!")
            run_monocular(
                os.path.normpath(args.root_dir),
                os.path.normpath(args.out_dir_prefix),
                dataset,
                monocular_cam_overrides,
            )
        dataset_post_process(
            os.path.normpath(args.root_dir), os.path.normpath(args.out_dir_prefix), dataset, monocular_cam_overrides
        )
    if args.run_kinetic:
        print("Running kinetic dataset")
        if args.clean:
            print("Generating the 3D reconstructions...this will take a while!")
            run_kinetic(os.path.normpath(args.root_dir), os.path.normpath(args.out_dir_prefix))
        kinetic_analysis(os.path.normpath(args.root_dir), os.path.normpath(args.out_dir_prefix))
    if args.run_grid_search:
        print("Running data-driven grid search procedure")
        run_grid_search(os.path.normpath(args.root_dir), os.path.normpath(args.out_dir_prefix), dataset)
    if args.run_analysis:
        print("Running monocular analysis")
        if args.clean:
            print("Generating the 3D reconstructions...this will take a while!")
            run_monocular_all(os.path.normpath(args.root_dir), os.path.normpath(args.out_dir_prefix), dataset)
        distance_vs_error(
            os.path.normpath(args.root_dir),
            os.path.normpath(args.out_dir_prefix),
            dataset,
            relative=False,
            remove_outliers=False,
        )
    if args.run_data_driven_ablation_study:
        print("Running data driven ablation study")
        run_data_driven_ablation_study(os.path.normpath(args.root_dir), os.path.normpath(args.out_dir_prefix), dataset)
    if args.run_physics_based_ablation_study:
        print("Running physics based ablation study")
        run_physics_based_ablation_study(
            os.path.normpath(args.root_dir), os.path.normpath(args.out_dir_prefix), dataset
        )
