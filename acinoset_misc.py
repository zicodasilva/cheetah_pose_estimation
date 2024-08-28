from dataclasses import dataclass
import os
from typing import Iterable, List, Tuple, Dict, Optional, Union, Any, cast
import numpy as np
import sympy as sp
import pandas as pd
import pickle
from glob import glob
from errno import ENOENT
import json
import pyomo.environ as pyo
import cv2 as cv
import shared.physical_education as pe
from scipy import signal
from scipy.interpolate import UnivariateSpline
from acinoset_models import MotionModel, PoseModelGMM
from common.py_utils import data_ops

from matplotlib import rcParams, cycler

rcParams.update({"figure.autolayout": True})
rcParams.update({"font.size": 18})
rcParams.update({"font.serif": "Times New Roman"})
rcParams.update({"font.family": "serif"})

# Universal colour palatte for results.
plot_color = {
    "orange": "#FF6400",
    "blue": "#6699FF",
    "charcoal": "#5A5A5A",
    "gray": "#808080",
    "green": "#2E8B57",
    "red": "#DC143C",
}

rcParams["axes.prop_cycle"] = cycler(
    color=[plot_color["charcoal"], plot_color["orange"], plot_color["red"], plot_color["green"]]
)


@dataclass
class TrajectoryParams:
    data_dir: str
    start_frame: int
    end_frame: int
    total_length: int
    dlc_thresh: float
    sync_offset: Optional[List[Dict]]
    hand_labeled_data: bool
    kinetic_dataset: bool
    enable_shutter_delay_estimation: bool
    enable_ppms: bool


@dataclass
class Scene:
    scene_fpath: str
    k_arr: List[float]
    d_arr: List[float]
    r_arr: List[List[float]]
    t_arr: List[float]
    cam_res: List[int]
    fps: float
    n_cams: int
    cam_idx: Optional[int] = None


@dataclass
class SimpleLinearModel:
    m: float
    c: float

    def __init__(self, pts: Union[List, np.ndarray], verbose: bool = False) -> None:
        x_coords, y_coords = zip(*pts)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        self.m, self.c = np.linalg.lstsq(A, y_coords, rcond=None)[0]  # type: ignore
        if verbose:
            print("Line Solution is y = {m:.3f}x + {c:.3f}".format(m=self.m, c=self.c))

    def predict(self, x: float) -> float:
        return self.m * x + self.c


def bound_value(val: float, slack_percentage: float):
    if val > 0:
        return ((1 - slack_percentage) * val, (1 + slack_percentage) * val)
    elif val < 0:
        return ((1 + slack_percentage) * val, (1 - slack_percentage) * val)
    else:
        return (-1 * slack_percentage, 1 * slack_percentage)


def rmse(predictions, targets):
    differences = predictions - targets
    differences_squared = differences**2
    mean_of_differences_squared = np.nanmean(differences_squared.flatten())
    rmse_val = np.sqrt(mean_of_differences_squared)
    return rmse_val


def create_foot_contraints(robot: pe.system.System3D):
    m = cast(pyo.ConcreteModel, robot.m)
    m.HFR_height = pyo.Var(m.fe, m.cp, name="HFR_height", bounds=(0, None))
    m.HFL_height = pyo.Var(m.fe, m.cp, name="HFL_height", bounds=(0, None))
    m.HBR_height = pyo.Var(m.fe, m.cp, name="HBR_height", bounds=(0, None))
    m.HBL_height = pyo.Var(m.fe, m.cp, name="HBL_height", bounds=(0, None))
    HFR_pos_func = pe.utils.lambdify_EOM(robot["HFR"].bottom_I, robot.sp_variables)
    HFL_pos_func = pe.utils.lambdify_EOM(robot["HFL"].bottom_I, robot.sp_variables)
    HBR_pos_func = pe.utils.lambdify_EOM(robot["HBR"].bottom_I, robot.sp_variables)
    HBL_pos_func = pe.utils.lambdify_EOM(robot["HBL"].bottom_I, robot.sp_variables)

    ncp = len(cast(Any, m).cp)

    def add_constraints(name: str, func, indexes):
        setattr(robot.m, name, pyo.Constraint(*indexes, rule=func))

    def def_foot_height(m, fe, cp):  # foot height above z == 0 (xy-plane)
        if fe == 1 and cp < ncp:
            return pyo.Constraint.Skip
        return m.HFR_height[fe, cp] == HFR_pos_func[2](robot.pyo_variables[fe, cp])

    add_constraints("HFR_height_constr", def_foot_height, (m.fe, m.cp))

    def def_foot_height1(m, fe, cp):  # foot height above z == 0 (xy-plane)
        if fe == 1 and cp < ncp:
            return pyo.Constraint.Skip
        return m.HFL_height[fe, cp] == HFL_pos_func[2](robot.pyo_variables[fe, cp])

    add_constraints("HFL_height_constr", def_foot_height1, (m.fe, m.cp))

    def def_foot_height2(m, fe, cp):  # foot height above z == 0 (xy-plane)
        if fe == 1 and cp < ncp:
            return pyo.Constraint.Skip
        return m.HBR_height[fe, cp] == HBR_pos_func[2](robot.pyo_variables[fe, cp])

    add_constraints("HBR_height_constr", def_foot_height2, (m.fe, m.cp))

    def def_foot_height3(m, fe, cp):  # foot height above z == 0 (xy-plane)
        if fe == 1 and cp < ncp:
            return pyo.Constraint.Skip
        return m.HBL_height[fe, cp] == HBL_pos_func[2](robot.pyo_variables[fe, cp])

    add_constraints("HBL_height_constr", def_foot_height3, (m.fe, m.cp))


def create_camera_contraints(
    robot: pe.system.System3D,
    params: TrajectoryParams,
    scene: Scene,
    dlc_dir: str,
    position_funcs,
    hand_labeled_data: bool = False,
    kinetic_dataset: bool = False,
):
    q, _, _ = robot.get_state_vars()
    markers = get_markers()
    m = cast(pyo.ConcreteModel, robot.m)

    def pt3d_to_x2d(x, y, z, K, D, R, t):
        return pt3d_to_2d(x, y, z, K, D, R, t)[0] if kinetic_dataset else pt3d_to_2d_fisheye(x, y, z, K, D, R, t)[0]

    def pt3d_to_y2d(x, y, z, K, D, R, t):
        return pt3d_to_2d(x, y, z, K, D, R, t)[1] if kinetic_dataset else pt3d_to_2d_fisheye(x, y, z, K, D, R, t)[1]

    proj_funcs = [pt3d_to_x2d, pt3d_to_y2d]
    L = len(markers)  # number of dlc labels per frame
    # C = len(scene.k_arr) if scene.cam_idx is None else 1  # number of cameras
    m.P = pyo.RangeSet(len(set(cast(Any, q))))
    m.L = pyo.RangeSet(L)
    # number of cameras. Note, this only works if cam index you select is within a consecutive set of cameras.
    if scene.cam_idx is None:
        m.C = pyo.RangeSet(len(scene.k_arr))
    else:
        m.C = pyo.RangeSet(scene.cam_idx + 1, scene.cam_idx + 1)
    # Dimensionality of measurements
    m.D2 = pyo.RangeSet(2)
    m.D3 = pyo.RangeSet(3)
    # Number of pairwise terms to include + the base measurement.
    m.W = pyo.RangeSet(3 if params.enable_ppms else 1)
    m.pose = pyo.Var(m.fe, m.L, m.D3)
    m.slack_meas = pyo.Var(m.fe, m.C, m.L, m.D2, m.W, initialize=0.0)
    if scene.cam_idx is None and params.enable_shutter_delay_estimation:
        m.shutter_delay = pyo.Var(m.C, initialize=0.0, bounds=(-m.hm0.value, m.hm0.value))
    index_dict = get_dlc_marker_indices()
    pair_dict = get_pairwise_graph()
    R_pw, Q = get_uncertainty_models()
    if kinetic_dataset:
        R_pw[:] = 7
    base_data = {}
    pw_data = {}
    cam_idx = 0
    # Extract the frame offsets to synchronise all cameras.
    sync_offset_arr = [0] * scene.n_cams
    if params.sync_offset is not None:
        for offset in params.sync_offset:
            sync_offset_arr[offset["cam"]] = offset["frame"]
    df_paths = sorted(glob(os.path.join(dlc_dir, "*.h5")))
    assert scene.n_cams == len(df_paths), f"# of dlc .h5 files != # of cams in {scene.n_cams}_cam_scene_sba.json"
    for path in df_paths:
        # Pairwise correspondence data.
        h5_filename = os.path.basename(path)
        if params.enable_ppms:
            pw_data[cam_idx] = data_ops.load_pickle(
                os.path.join(dlc_dir + "_pw", f"{h5_filename[:4]}DLC_resnet152_CheetahOct14shuffle4_650000.pickle")
            )
        df_temp = pd.read_hdf(os.path.join(dlc_dir, h5_filename))
        base_data[cam_idx] = list(df_temp.to_numpy())
        cam_idx += 1

    # ======= WEIGHTS =======
    def init_meas_weights(_, n, c, l, w):
        # Determine if the current measurement is the base prediction or a pairwise prediction.
        cam_idx = c - 1
        marker = markers[l - 1]
        if w < 2:
            base = index_dict[marker]
            if hand_labeled_data:
                likelihoods = np.ones((len(markers) + 1, 1)).flatten()
                x_pixel = base_data[cam_idx][(n - 1) + params.start_frame][0::2][base]
                y_pixel = base_data[cam_idx][(n - 1) + params.start_frame][1::2][base]
                if x_pixel == np.nan and y_pixel == np.nan:
                    likelihoods[base] = 0.0
            else:
                likelihoods = base_data[cam_idx][(n - 1) + params.start_frame - sync_offset_arr[cam_idx]][2::3]
        else:
            base = pair_dict[marker][w - 2]
            values = pw_data[cam_idx][(n - 1) + params.start_frame - sync_offset_arr[cam_idx]]
            likelihoods = values["pose"][2::3]
        # Filter measurements based on DLC threshold.
        # This does ensures that badly predicted points are not considered in the objective function.
        return 1 / R_pw[w - 1][l - 1] if likelihoods[base] > params.dlc_thresh else 0.0

    m.meas_err_weight = pyo.Param(m.fe, m.C, m.L, m.W, initialize=init_meas_weights, mutable=True)
    m.model_err_weight = pyo.Param(m.P, initialize=lambda m, p: 1 / Q[p - 1] if Q[p - 1] != 0.0 else 0.0)

    # ===== PARAMETERS =====
    def init_measurements(_, n, c, l, d2, w):
        # Determine if the current measurement is the base prediction or a pairwise prediction.
        cam_idx = c - 1
        marker = markers[l - 1]
        if w < 2:
            base = index_dict[marker]
            if hand_labeled_data:
                val = base_data[cam_idx][(n - 1) + params.start_frame][d2 - 1 :: 2]
            else:
                val = base_data[cam_idx][(n - 1) + params.start_frame - sync_offset_arr[cam_idx]][d2 - 1 :: 3]

            return val[base] if val[base] != np.nan else 0.0
        else:
            values = pw_data[cam_idx][(n - 1) + params.start_frame - sync_offset_arr[cam_idx]]
            val = values["pose"][d2 - 1 :: 3]
            base = pair_dict[marker][w - 2]
            val_pw = values["pws"][:, :, :, d2 - 1]
            return val[base] + val_pw[0, base, index_dict[marker]]

    m.meas = pyo.Param(m.fe, m.C, m.L, m.D2, m.W, initialize=init_measurements)
    print("Get position variables to build pose constraint")
    ncp = len(cast(Any, m).cp)
    var_list = []
    for fe in cast(Iterable, m.fe):
        q_list = []
        l_list = []
        for link in robot.links:
            pyo_vars = robot[link.name].get_pyomo_vars(fe, ncp)
            q_list += pyo_vars[: len(robot[link.name].q)]
            l_list.append(robot[link.name].pyomo_params["length"])
        var_list.append(q_list + l_list)
    # 3D POSE
    m.pose_constraint = pyo.Constraint(
        m.fe, m.L, m.D3, rule=lambda m, fe, l, d3: position_funcs[l - 1][d3 - 1](var_list[fe - 1]) == m.pose[fe, l, d3]
    )
    print("Add measurement model")
    # Set first camera as reference for shutter delay estimation between cameras.
    if hasattr(m, "shutter_delay"):
        m.shutter_delay[1].fix(0)

    # 2D rerojection constraint.
    def measurement_constraints(m, n, c, l, d2, w):
        # project
        tau = m.shutter_delay[c] if hasattr(m, "shutter_delay") else 0.0
        K, D, R, t = scene.k_arr[c - 1], scene.d_arr[c - 1], scene.r_arr[c - 1], scene.t_arr[c - 1]
        # x, y, z = m.pose[n, l, 1], m.pose[n, l, 2], m.pose[n, l, 3]
        x = m.pose[n, l, 1] + robot["base"]["dq"][n, 1, "x"] * tau + robot["base"]["ddq"][n, 1, "x"] * (tau**2)
        y = m.pose[n, l, 2] + robot["base"]["dq"][n, 1, "y"] * tau + robot["base"]["ddq"][n, 1, "y"] * (tau**2)
        z = m.pose[n, l, 3] + robot["base"]["dq"][n, 1, "z"] * tau + robot["base"]["ddq"][n, 1, "z"] * (tau**2)
        return proj_funcs[d2 - 1](x, y, z, K, D, R, t) - m.meas[n, c, l, d2, w] - m.slack_meas[n, c, l, d2, w] == 0.0

    m.measurement = pyo.Constraint(m.fe, m.C, m.L, m.D2, m.W, rule=measurement_constraints)


def add_linear_motion_model(robot: pe.system.System3D, window_size: int, lasso: bool, data_dir: str):
    m = cast(pyo.ConcreteModel, robot.m)
    # Train motion model and make predictions with a predefined window size.
    num_vars = 28
    window_time = 1
    window_buf = window_size * window_time
    motion_model = MotionModel(
        os.path.join(".", "models", "data-driven", "dataset_full_pose.h5"),
        num_vars,
        window_size=window_size,
        window_time=window_time,
        lasso=lasso,
    )
    pred_var = motion_model.error_variance
    pose_mask = get_relative_angle_mask()
    m.R = pyo.RangeSet(num_vars)
    m.motion_err_weight = pyo.Param(m.R, initialize=lambda m, p: 1 / pred_var[p - 1] if pred_var[p - 1] != 0 else 0.0)
    m.slack_motion = pyo.Var(m.fe, m.cp, m.R, initialize=0.0)
    # Incorporate the prediction model as constraints.
    X = np.array(
        [
            np.array(pe.utils.flatten(get_relative_angles(robot, fe, cp)))[pose_mask]
            for fe, cp in robot.indices(one_based=True)
        ]
    )
    df = data_ops.series_to_supervised(X, n_in=window_size, n_step=window_time)
    X_in = df.to_numpy()[:, 0 : (num_vars * window_size)]
    y_pred = motion_model.predict(X_in, matrix=True)

    def motion_constr(m_temp: pyo.ConcreteModel, fe, cp, p):
        if fe > window_buf:
            x = np.array(pe.utils.flatten(get_relative_angles(robot, fe, cp)))[pose_mask]
            return x[p - 1] - y_pred[fe - window_buf - 1, p - 1] - m.slack_motion[fe, cp, p] == 0
        else:
            return pyo.Constraint.Skip

    print("Add LR motion model")
    setattr(m, "monocular_motion_constr", pyo.Constraint(m.fe, m.cp, m.R, rule=motion_constr))

    slack_motion_err = 0.0
    for fe in cast(Iterable, m.fe):
        for cp in cast(Iterable, m.cp):
            for p in cast(Iterable, m.R):
                slack_motion_err += m.motion_err_weight[p] * m.slack_motion[fe, cp, p] ** 2

    return slack_motion_err, motion_model


def init_foot_height(robot: pe.system.System3D):
    m = cast(pyo.ConcreteModel, robot.m)
    ncp = len(cast(Any, m).cp)
    for foot in pe.foot.feet(robot):
        for fe in cast(Iterable, m.fe):
            foot["foot_height"][fe, ncp].value = pyo.value(foot.foot_pos_func[2](robot.pyo_variables[fe, ncp]))


def init_foot_velocity(robot: pe.system.System3D) -> np.ndarray:
    m = cast(pyo.ConcreteModel, robot.m)
    ncp = len(cast(Any, m).cp)
    # 4 - four feet, and 3 - 3D space.
    ret = np.empty((len(cast(Any, m).fe), 4, 3))
    data: List[List[float]] = [
        [cast(Any, v).value for v in robot.pyo_variables[fe, ncp]] for fe in cast(Iterable, m.fe)
    ]
    for j, foot in enumerate(cast(List[pe.foot.Foot3D], pe.foot.feet(robot))):
        func = pe.utils.lambdify_EOM(foot.Pb_I_vel, robot.sp_variables)
        for fe in cast(Iterable, m.fe):
            for i in range(3):
                ret[fe - 1, j, i] = func[i](data[fe - 1])

    return ret


def init_3d_pose(robot: pe.system.System3D, position_funcs):
    m = cast(pyo.ConcreteModel, robot.m)
    q_init = pe.utils.get_vals(robot["base"].pyomo_vars["q"], (robot["base"].pyomo_sets["q_set"],)).squeeze()
    link_lengths = [robot["base"].length]
    for link in robot.links[1:]:
        q_init = np.concatenate(
            (q_init, pe.utils.get_vals(link.pyomo_vars["q"], (link.pyomo_sets["q_set"],)).squeeze(axis=1)), axis=1
        )
        link_lengths.append(link.length)
    link_lengths = np.tile(link_lengths, q_init.shape[0]).reshape(q_init.shape[0], -1)
    states = np.concatenate((q_init, link_lengths), axis=1)
    for fe in cast(Iterable, m.fe):
        for l in cast(Iterable, m.L):
            for d3 in cast(Iterable, m.D3):
                cast(Any, m).pose[fe, l, d3].value = position_funcs[l - 1][d3 - 1](states[fe - 1, :])


def create_trajectory_estimate(
    robot: pe.system.System3D, params: TrajectoryParams, scene: Scene, kinetic_dataset: bool
):
    # load DLC data
    dlc_points_fpaths = sorted(
        glob(os.path.join(params.data_dir, "dlc" if not params.hand_labeled_data else "dlc_hand_labeled", "*.h5"))
    )
    # load measurement dataframe (pixels, likelihood)
    points_2d_df = load_dlc_points_as_df(dlc_points_fpaths, hand_labeled=params.hand_labeled_data, verbose=False)
    if not params.hand_labeled_data:
        points_2d_df = points_2d_df[
            points_2d_df["likelihood"] > params.dlc_thresh
        ]  # ignore points with low likelihood
    # Ensure that the cameras are synchonised.
    if params.sync_offset is not None:
        for offset in params.sync_offset:
            points_2d_df.loc[points_2d_df["camera"] == offset["cam"], "frame"] += offset["frame"]
    # Add init of state vector.
    if kinetic_dataset:
        # Only consider the near side cameras for the initial trajectory.
        points_2d_df = points_2d_df[points_2d_df["camera"] < 2]
    if scene.cam_idx is None:
        points_3d_df = get_pairwise_3d_points_from_df(
            points_2d_df,
            scene.k_arr,
            scene.d_arr,
            scene.r_arr,
            scene.t_arr,
            triangulate_points if kinetic_dataset else triangulate_points_fisheye,
        )
        spine_pts = points_3d_df[points_3d_df["marker"] == "spine"][["frame", "x", "y", "z"]].values
    else:
        points_2d_df = points_2d_df[points_2d_df["camera"] == scene.cam_idx]
        spine_pts_2d = points_2d_df.query("marker == 'spine'")[["frame", "x", "y"]].to_numpy(dtype=np.float32)
        spine_pts = triangulate_points_single_img(
            spine_pts_2d[:, 1:],
            3,
            scene.k_arr[scene.cam_idx],
            scene.d_arr[scene.cam_idx],
            scene.r_arr[scene.cam_idx],
            scene.t_arr[scene.cam_idx],
        ).T
        spine_pts = np.c_[spine_pts_2d[:, 0], spine_pts]
    spine_pts[:, 1] = spine_pts[:, 1] + cast(float, robot["base"].length) / 2.0
    if params.hand_labeled_data:
        num_frames = points_2d_df["frame"].max() + 1
        frame_range = np.arange(num_frames, dtype=int)
        w = np.isnan(spine_pts[:, 1])
        spine_pts[w, 1] = 0.0
        x_est = np.array(
            UnivariateSpline(frame_range, spine_pts[:, 1], w=~w, k=1 if kinetic_dataset else 3)(frame_range)
        )
        spine_pts[w, 2] = 0.0
        y_est = np.array(
            UnivariateSpline(frame_range, spine_pts[:, 2], w=~w, k=1 if kinetic_dataset else 3)(frame_range)
        )
        spine_pts[w, 3] = 0.0
        z_est = np.array(
            UnivariateSpline(frame_range, spine_pts[:, 3], w=~w, k=1 if kinetic_dataset else 3)(frame_range)
        )
    else:
        frame_est = np.arange(params.end_frame)
        traj_est_x = UnivariateSpline(spine_pts[:, 0], spine_pts[:, 1], k=1 if kinetic_dataset else 3)
        traj_est_y = UnivariateSpline(spine_pts[:, 0], spine_pts[:, 2], k=1 if kinetic_dataset else 3)
        traj_est_z = UnivariateSpline(spine_pts[:, 0], spine_pts[:, 3], k=1 if kinetic_dataset else 3)
        x_est = np.array(traj_est_x(frame_est))
        y_est = np.array(traj_est_y(frame_est))
        z_est = np.array(traj_est_z(frame_est))
    # Calculate the initial yaw.
    dx_est = np.diff(x_est) * scene.fps
    dy_est = np.diff(y_est) * scene.fps
    psi_est = np.arctan2(dy_est, dx_est)
    # Duplicate the last heading estimate as the difference calculation returns N-1.
    psi_est = np.pi + np.append(psi_est, [psi_est[-1]])  # TODO: This assumes the cheetah is running +x direction.

    return x_est, y_est, z_est, psi_est


def measurement_cost(robot: pe.system.System3D, hand_labeled_data: bool = False, kinetic_dataset: bool = False):
    m = cast(pyo.ConcreteModel, robot.m)
    slack_meas_err = 0.0
    cam_uncertainty_multiplier = (
        [1.0, 1.0, 0.6, 0.6] if kinetic_dataset else [1.0] * 6
    )  # Assume a maximum 6-camera setup.
    for fe in cast(Iterable, m.fe):
        # Measurement Error
        for l in cast(Iterable, m.L):
            for c in cast(Iterable, m.C):
                for d2 in cast(Iterable, m.D2):
                    for w in cast(Iterable, m.W):
                        if hand_labeled_data:
                            slack_meas_err += (
                                cast(Any, m).meas_err_weight[fe, c, l, w] * cast(Any, m).slack_meas[fe, c, l, d2, w]
                            ) ** 2
                        else:
                            slack_meas_err += redescending_loss(
                                (cam_uncertainty_multiplier[c - 1] * cast(Any, m).meas_err_weight[fe, c, l, w])
                                * cast(Any, m).slack_meas[fe, c, l, d2, w],
                                3,
                                10,
                                20,
                            )

    return slack_meas_err


def get_relative_angles(input: Union[pe.system.System3D, np.ndarray], fe: int, cp: int = 1, var: str = "q"):
    if isinstance(input, pe.system.System3D):
        # Get relative angles using pyomo variables.
        angles = ["phi", "theta", "psi"]
        base = [input["base"][var][fe, cp, q] for q in input["base"].pyomo_sets["q_set"]]
        body_F = [input["bodyF"][var][fe, cp, q] - input["base"][var][fe, cp, q] for q in angles]
        neck = [input["neck"][var][fe, cp, q] - input["bodyF"][var][fe, cp, q] for q in angles]
        tail0 = [input["base"][var][fe, cp, q] - input["tail0"][var][fe, cp, q] for q in angles]
        tail1 = [input["tail0"][var][fe, cp, q] - input["tail1"][var][fe, cp, q] for q in angles]
        ufr = [input["bodyF"][var][fe, cp, q] - input["UFR"][var][fe, cp, q] for q in angles]
        ufl = [input["bodyF"][var][fe, cp, q] - input["UFL"][var][fe, cp, q] for q in angles]
        ubr = [input["base"][var][fe, cp, q] - input["UBR"][var][fe, cp, q] for q in angles]
        ubl = [input["base"][var][fe, cp, q] - input["UBL"][var][fe, cp, q] for q in angles]
        lfr = [input["UFR"][var][fe, cp, q] - input["LFR"][var][fe, cp, q] for q in angles]
        lfl = [input["UFL"][var][fe, cp, q] - input["LFL"][var][fe, cp, q] for q in angles]
        lbr = [input["UBR"][var][fe, cp, q] - input["LBR"][var][fe, cp, q] for q in angles]
        lbl = [input["UBL"][var][fe, cp, q] - input["LBL"][var][fe, cp, q] for q in angles]
        hfr = [input["LFR"][var][fe, cp, q] - input["HFR"][var][fe, cp, q] for q in angles]
        hfl = [input["LFL"][var][fe, cp, q] - input["HFL"][var][fe, cp, q] for q in angles]
        hbr = [input["LBR"][var][fe, cp, q] - input["HBR"][var][fe, cp, q] for q in angles]
        hbl = [input["LBL"][var][fe, cp, q] - input["HBL"][var][fe, cp, q] for q in angles]
    else:
        # Get relative angles from numpy array.
        base = list(input[fe, 0:6])
        body_F = list(input[fe, 6:9] - input[fe, 3:6])
        neck = list(input[fe, 9:12] - input[fe, 6:9])
        tail0 = list(input[fe, 3:6] - input[fe, 12:15])
        tail1 = list(input[fe, 12:15] - input[fe, 15:18])
        ufr = list(input[fe, 6:9] - input[fe, 27:30])
        ufl = list(input[fe, 6:9] - input[fe, 18:21])
        ubr = list(input[fe, 3:6] - input[fe, 42:45])
        ubl = list(input[fe, 3:6] - input[fe, 36:39])
        lfr = list(input[fe, 27:30] - input[fe, 30:33])
        lfl = list(input[fe, 18:21] - input[fe, 21:24])
        lbr = list(input[fe, 42:45] - input[fe, 45:48])
        lbl = list(input[fe, 36:39] - input[fe, 39:42])
        hfr = list(input[fe, 30:33] - input[fe, 33:36])
        hfl = list(input[fe, 21:24] - input[fe, 24:27])
        hbr = list(input[fe, 45:48] - input[fe, 51:54])
        hbl = list(input[fe, 39:42] - input[fe, 48:51])

    return [base, body_F, neck, tail0, tail1, ufl, lfl, hfl, ufr, lfr, hfr, ubl, lbl, ubr, lbr, hbl, hbr]


def kinematic_cost(robot: pe.system.System3D, desired_q: np.ndarray, total_mass: float):
    slack_kinematcs_err = 0.0
    M = [
        10,  # x_{base_bodyB}
        10,  # y_{base_bodyB}
        10,  # z_{base_bodyB}
        5,  # \phi_{base_bodyB}
        5,  # \theta_{base_bodyB}
        5,  # \psi_{base_bodyB}
        0,  # \phi_{body_F}
        5,  # \theta_{body_F}
        5,  # \psi_{body_F}
        0,  # \phi_{neck}
        2,  # \theta_{neck}
        2,  # \psi_{neck}
        0,  # \phi_{tail0}
        5,  # \theta_{tail0}
        5,  # \psi_{tail0}
        0,  # \phi_{tail1}
        5,  # \theta_{tail1}
        5,  # \psi_{tail1}
        0,  # \phi_{UFL}
        5,  # \theta_{UFL}
        0,  # \psi_{UFL}
        0,  # \phi_{LFL}
        2,  # \theta_{LFL}
        0,  # \psi_{LFL}
        0,  # \phi_{HFL}
        1,  # \theta_{HFL}
        0,  # \psi_{HFL}
        0,  # \phi_{UFR}
        5,  # \theta_{UFR}
        0,  # \psi_{UFR}
        0,  # \phi_{LFR}
        2,  # \theta_{LFR}
        0,  # \psi_{LFR}
        0,  # \phi_{HFR}
        1,  # \theta_{HFR}
        0,  # \psi_{HFR}
        0,  # \phi_{UBL}
        5,  # \theta_{UBL}
        0,  # \psi_{UBL}
        0,  # \phi_{LBL}
        2,  # \theta_{LBL}
        0,  # \psi_{LBL}
        0,  # \phi_{UBR}
        5,  # \theta_{UBR}
        0,  # \psi_{UBR}
        0,  # \phi_{LBR}
        2,  # \theta_{LBR}
        0,  # \psi_{LBR}
        0,  # \phi_{HBL}
        1,  # \theta_{HBL}
        0,  # \psi_{HBL}
        0,  # \phi_{HBR}
        1,  # \theta_{HBR}
        0,  # \psi_{HBR}
    ]
    # M = np.ones(54)
    # M[0:6] = [3] * 6
    for fe, cp in robot.indices(one_based=True):
        p = 0
        q_gt = pe.utils.flatten(get_relative_angles(desired_q, fe - 1, cp))
        q_est = pe.utils.flatten(get_relative_angles(robot, fe, cp))
        for q_d, q in zip(q_gt, q_est):
            slack_kinematcs_err += M[p] * (q_d - q) ** 2
            p += 1
    return slack_kinematcs_err


def motion_smoothing_cost(robot: pe.system.System3D, fps: float):
    # Minimise the linear joint acceleration.
    energy_acc = 0.0
    m = cast(Any, robot.m)
    nfe = len(m.fe)
    for fe in m.fe:
        if fe < nfe - 2:
            for l in m.L:
                for d in m.D3:
                    pt_acc = fps**2 * (m.pose[fe + 2, l, d] - 2 * m.pose[fe + 1, l, d] + m.pose[fe, l, d])
                    energy_acc += pt_acc**2

    return energy_acc


def change_in_torque_squared_cost(robot: pe.system.System3D):
    m = cast(Any, robot.m)
    nfe = len(m.fe)
    dtau = 0.0
    for motor in pe.motor.torques(robot):
        Tc = motor.pyomo_vars["Tc"]
        Tc_set = motor.pyomo_sets["Tc_set"]
        for fe in m.fe:
            if fe < nfe - 1:
                for idx in Tc_set:
                    dtau += ((Tc[fe + 1, idx] - Tc[fe, idx]) / m.hm0) ** 2

    return dtau


def eom_slack_cost(robot: pe.system.System3D):
    slack_model_err = 0.0
    for fe, cp in robot.indices(one_based=True):
        for i in range(len(robot.eom)):
            slack_model_err += robot.m.slack_eom[fe, cp, i] ** 2
    return slack_model_err


def constant_acc_cost(robot: pe.system.System3D):
    m = cast(pyo.ConcreteModel, robot.m)

    def add_constant_acc_constraint(m: pyo.ConcreteModel, link: pe.links.Link3D):
        var_name = f"{link.name}_acc_model"
        ddq = link.pyomo_vars["ddq"]
        setattr(m, var_name, pyo.Var(m.fe, m.cp, link.pyomo_sets["q_set"]))
        setattr(
            m,
            f"{var_name}_constr",
            pyo.Constraint(
                m.fe,
                m.cp,
                link.pyomo_sets["q_set"],
                rule=lambda m_temp, fe, cp, q: (
                    ddq[fe, cp, q] == ddq[fe - 1, cp, q] + getattr(m_temp, var_name)[fe, cp, q]
                    if fe > 1
                    else pyo.Constraint.Skip
                ),
            ),
        )

    print("Add constant acceleration model")
    for link in robot.links:
        add_constant_acc_constraint(m, link)

    print("Add constant acceleration cost")
    slack_model_err = 0.0
    for fe in cast(Iterable, m.fe):
        # Model Error
        for cp in cast(Iterable, m.cp):
            p = 1
            for link in robot.links:
                var_name = f"{link.name}_acc_model"
                for q in link.pyomo_sets["q_set"]:
                    slack_model_err += cast(Any, m).model_err_weight[p] * getattr(robot.m, var_name)[fe, cp, q] ** 2
                    p += 1

    return slack_model_err


def gmm_pose_cost(robot: pe.system.System3D, n_comps: int, data_dir: str):
    from pyomo.core.expr.current import log as pyomo_log, exp

    # Add GMM model if we are performing monocular reconstruction.
    pose_gmm = PoseModelGMM(
        os.path.join(".", "models", "data-driven", "dataset_full_pose.h5"),
        num_vars=28,
        ext_dim=6,
        n_comps=n_comps,
    )

    def norm_pdf_multivariate(x: np.ndarray, mu: np.ndarray, cov: np.ndarray):
        part1 = 1 / (((2 * np.pi) ** (len(x) / 2)) * (np.linalg.det(cov) ** (1 / 2)))
        part2 = (-1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
        return part1 * exp(part2)

    slack_pose_err = 0.0
    for fe, cp in robot.indices(one_based=True):
        x = np.array(pe.utils.flatten(get_relative_angles(robot, fe, cp)))[get_relative_angle_mask()][6:]
        slack_pose_err += -pyomo_log(
            sum(
                [
                    w * norm_pdf_multivariate(x, mu, cov)
                    for w, mu, cov in zip(pose_gmm.gmm.weights_, pose_gmm.gmm.means_, pose_gmm.gmm.covariances_)
                ]
            )
            + 1e-12
        )
        # TODO: Need to double check whether this is valid? It seems to be taking the log likelihood of each mixture and summing together.
        # slack_pose_err += sum([
        #     w * (1 / 2) * ((x - mu).T.dot(np.linalg.inv(cov))).dot((x - mu))
        #     for w, mu, cov in zip(pose_gmm.gmm.weights_, pose_gmm.gmm.means_, pose_gmm.gmm.covariances_)
        # ])

    return slack_pose_err, pose_gmm


def remove_dc_offset(x, num_samples: int = 100):
    offset = np.mean(x[:num_samples], axis=0)
    return x - offset


def get_com(robot: pe.system.System3D, scene: Scene) -> Tuple[np.ndarray, np.ndarray]:
    m = cast(pyo.ConcreteModel, robot.m)
    ncp = len(cast(Any, m.cp))
    data: List[List[float]] = [
        [cast(Any, v).value for v in robot.pyo_variables[fe, ncp]] for fe in cast(Iterable, m.fe)
    ]
    # TODO: This should be placed inside the link and created at the same time as the link.
    pos_funcs = [pe.utils.lambdify_EOM(link.Pb_I, robot.sp_variables) for link in robot.links]
    total_mass = sum(cast(float, link.mass) for link in robot.links)
    com_position = np.empty((len(data), 3))
    for idx, d in enumerate(data):
        com_pos = [0.0] * 3
        for i, link in enumerate(robot.links):
            for j, f in enumerate(pos_funcs[i]):
                com_pos[j] += cast(float, link.mass) * f(d)
        com_pos = np.asarray(com_pos)
        com_pos *= 1 / total_mass
        com_position[idx, :] = com_pos

    # Return the COM position and velocities over the trajectory.
    return com_position, (com_position[1:, :] - com_position[:-1, :]) * scene.fps


def contact_detection(
    robot: pe.system.System3D, start_frame: int, speed: float, fps: float, data_dir: str, plot: bool = False
) -> Tuple[Dict, Dict]:
    # Determine a rough linear model for stance time vs speed of cheetah (from Penny Hudson"s paper).
    stance_time_model = SimpleLinearModel([[9.0, 0.09], [14.0, 0.06]])
    stance_time_fe = round(stance_time_model.predict(speed) * fps)
    mid_way = stance_time_fe // 2
    is_even = (stance_time_fe % 2) == 0
    # Init foot height.
    init_foot_height(robot)
    # Thresholds.
    height_threshold = 0.05
    # Init foot xy velocity.
    foot_vel = init_foot_velocity(robot)
    contacts = {}
    contacts_tmp = {}
    if plot:
        import matplotlib.pyplot as plt

        _ = plt.figure(figsize=(16, 9), dpi=120)
    idx = 0
    for i, foot in enumerate(pe.foot.feet(robot)):
        foot_height = pe.utils.get_vals(foot.pyomo_vars["foot_height"], tuple())
        if plot:
            ax = plt.subplot(2, 2, idx + 1)  # type: ignore
            ax.plot(foot_height, label="Height", color=plot_color["charcoal"], marker="o")
            ax.set_title(foot.name)
            ax.legend()
            ax2 = ax.twinx()
            ax2.plot(foot_vel[:, idx, 2], label="Vel", color=plot_color["orange"], marker="o")
            ax2.legend()
        arg_height_heuristic = np.where(foot_height[:, 0] < (pe.foot.Foot3D.ground_plane_height + height_threshold))
        arg_height_heuristic = group_by_consecutive_values(arg_height_heuristic[0])
        _, arg_vel_zero_crossings = positive_zero_crossings(foot_vel[:, idx, 2])
        idx += 1
        contacts[foot.name] = []
        contacts_tmp[foot.name] = []
        arg_min_height = -1
        for j, pos_foot_contact in enumerate(arg_height_heuristic):
            if len(arg_height_heuristic[j]) == 0:
                # No point in trying to determine a contact if there is nothing under the height heuristic.
                continue
            start_search = int(arg_min_height + 1)
            arg_min_height = find_minimum_foot_height(
                foot_height[:, 0],
                (start_search, arg_height_heuristic[j + 1][0] if j + 1 < len(arg_height_heuristic) else -1),
            )
            possible_contact_detected = np.intersect1d(pos_foot_contact, arg_vel_zero_crossings)
            is_contact = [arg_min_height + k not in possible_contact_detected for k in [-2, -1, 0, 1, 2]]
            if np.all(is_contact):
                # Update the arg_min_height to be at the end of the height heuristic so you do not detect the argmin of the same contact.
                arg_min_height = arg_height_heuristic[j][-1]
                continue
            start_idx = int(arg_min_height - mid_way)
            # start_idx = int(arg_height_heuristic[0][0])
            end_idx = int(arg_min_height + mid_way)
            # end_idx = start_idx + stance_time_fe
            arg_min_height = arg_height_heuristic[j][-1]
            if is_even:
                start_idx += 1
            if start_idx < 0:
                end_idx -= start_idx
                start_idx = 0
            if end_idx >= len(robot.m.fe):
                start_idx -= end_idx - len(robot.m.fe) - 1
                end_idx = len(robot.m.fe) - 1
            contacts[foot.name].append([start_frame + start_idx, start_frame + end_idx, i, "TBD"])
            # A second approach to contact detection where only the height threshold is used and no prior on the stance length and zero velocity.
            contacts_tmp[foot.name].append(
                [int(start_frame + pos_foot_contact[0]), int(start_frame + pos_foot_contact[-1]), i, "TBD"]
            )
        # Set None for feet where there is no contact found.
        if len(contacts[foot.name]) == 0:
            contacts[foot.name] = None
            contacts_tmp[foot.name] = None
    if plot:
        plt.show(block=False)  # type: ignore
    # Assuming a rotary gollap, trailing limb will correspond to the limb that is in contact first.
    # and the leading limb will be in contact sometime after.
    # TODO: When both limbs do not have a contact during the reconstruction, we can't tell which one is leading/trailing?
    # Forelimbs
    if contacts["HFL_foot"] != None and contacts["HFR_foot"] != None:
        if contacts["HFL_foot"][0][0] > contacts["HFR_foot"][0][0]:
            contacts["HFL_foot"][0][3] = "leading"
            contacts["HFR_foot"][0][3] = "trailing"
        else:
            contacts["HFL_foot"][0][3] = "trailing"
            contacts["HFR_foot"][0][3] = "leading"
    # Hindlimbs
    if contacts["HBL_foot"] != None and contacts["HBR_foot"] != None:
        if contacts["HBL_foot"][0][0] > contacts["HBR_foot"][0][0]:
            contacts["HBL_foot"][0][3] = "leading"
            contacts["HBR_foot"][0][3] = "trailing"
        else:
            contacts["HBL_foot"][0][3] = "trailing"
            contacts["HBR_foot"][0][3] = "leading"
    # Save data to contacts file.
    results = {"start_frame": start_frame, "end_frame": start_frame + len(cast(Any, robot).m.fe), "contacts": contacts}
    results2 = {
        "start_frame": start_frame,
        "end_frame": start_frame + len(cast(Any, robot).m.fe),
        "contacts": contacts_tmp,
    }
    grf_dir = os.path.join(data_dir, "grf")
    os.makedirs(grf_dir, exist_ok=True)
    with open(os.path.join(grf_dir, "autogen-contact.json"), "w", encoding="utf-8") as f:
        json.dump(results, f)
    f.close()
    with open(os.path.join(grf_dir, "autogen-contact-02.json"), "w", encoding="utf-8") as f:
        json.dump(results2, f)

    return contacts, contacts_tmp


def synth_grf_data(
    robot: pe.system.System3D,
    speed: float,
    direction: float,
    data_dir: str,
    contact_fname="autogen-contact.json",
    out_fname="data_synth",
) -> None:
    # Determine the linear models for the peak vertical force based on the speed of the cheetah (from Penny Hudson"s paper).
    from scipy import interpolate

    model_LFL = SimpleLinearModel([[9.0, 2.0], [15.0, 1.8]])
    model_LHL = SimpleLinearModel([[9.0, 2.1], [15.0, 2.6]])
    model_NLFL = SimpleLinearModel([[9.5, 2.1], [15.0, 2.0]])
    model_NLHL = SimpleLinearModel([[9.0, 1.7], [15.0, 2.5]])
    # 50% of the Fz peak will provide the cranial decceleration peak.
    # 50% of the Fx dec peak will provide the cranial acceleration peak.
    # Determine the contact times and generate the waveforms over that period.
    with open(os.path.join(data_dir, contact_fname), "r", encoding="utf-8") as f:
        contact_json = json.load(f)
    start_frame = contact_json["start_frame"]
    end_frame = contact_json["end_frame"]
    foot_contact_order = contact_json["contacts"]
    df_results = {}
    for foot in pe.foot.feet(robot):
        if (
            foot.name in foot_contact_order
            and foot_contact_order[foot.name] is not None
            and foot_contact_order[foot.name][0][1] < end_frame
        ):
            stance_start = 0
            start_idx = foot_contact_order[foot.name][0][0] - 1
            end_idx = foot_contact_order[foot.name][0][1] + 1
            if start_idx < start_frame:
                start_idx = start_frame
            if end_idx > end_frame:
                end_idx = end_frame
            stance_end = end_idx - start_idx
            peak_idx = stance_end // 2
            t = np.linspace(stance_start, stance_end, stance_end)
            Fz_peak, Fx_dec_peak, Fx_acc_peak = 0.0, 0.0, 0.0
            if "F" in foot.name and foot_contact_order[foot.name][0][3] == "leading":
                # LFL
                Fz_peak = model_LFL.predict(speed)
                Fx_dec_peak = direction * 0.5 * Fz_peak
                Fx_acc_peak = 0.5 * -Fx_dec_peak
            elif "F" in foot.name and foot_contact_order[foot.name][0][3] == "trailing":
                # NLFL
                Fz_peak = model_NLFL.predict(speed)
                Fx_dec_peak = direction * 0.5 * Fz_peak
                Fx_acc_peak = 0.5 * -Fx_dec_peak
            elif "B" in foot.name and foot_contact_order[foot.name][0][3] == "leading":
                # LHL
                Fz_peak = model_LHL.predict(speed)
                Fx_dec_peak = direction * 0.5 * Fz_peak
                Fx_acc_peak = 0.5 * -Fx_dec_peak
            elif "B" in foot.name and foot_contact_order[foot.name][0][3] == "trailing":
                # NLHL
                Fz_peak = model_NLHL.predict(speed)
                Fx_dec_peak = direction * 0.5 * Fz_peak
                Fx_acc_peak = 0.5 * -Fx_dec_peak
            synth_Fz = Fz_peak * np.sin(np.pi * (t / stance_end))
            Fx_control_pts = np.array(
                [
                    [stance_start, 0.0],
                    [peak_idx // 2, Fx_dec_peak],
                    [peak_idx, 0.0],
                    [peak_idx + (stance_end - peak_idx) // 2, Fx_acc_peak],
                    [stance_end, 0.0],
                ]
            )
            x = interpolate.InterpolatedUnivariateSpline(Fx_control_pts[:, 0], Fx_control_pts[:, 1], k=2)
            synth_Fx = x(t)
            Fz_synth = np.zeros(end_frame - start_frame)
            Fx_synth = np.zeros(end_frame - start_frame)
            Fy_synth = np.zeros(end_frame - start_frame)
            Fz_synth[start_idx - start_frame : end_idx - start_frame] = synth_Fz
            Fx_synth[start_idx - start_frame : end_idx - start_frame] = synth_Fx
            df_results[foot_contact_order[foot.name][0][2] - 1] = pd.DataFrame(
                np.array([Fx_synth, Fy_synth, Fz_synth]).T, columns=["Fx", "Fy", "Fz"]
            )
    df_synth = pd.concat(df_results.values(), keys=df_results.keys(), axis=0)
    df_synth.index.set_names(["force_plate", "frame"], inplace=True)
    out_fname = os.path.join(data_dir, f"{out_fname}.h5")
    df_synth.to_hdf(out_fname, "force_plate_data_df", format="table", mode="w")


def get_grf_profile(
    robot: pe.system.System3D,
    params: TrajectoryParams,
    direction: float,
    scale_forces_by: float,
    out_dir_prefix: Optional[str] = None,
    synthetic_data: bool = False,
) -> Tuple[Dict, Dict]:
    # Note, this assumes a single stride! So it takes the first value for each foot in the contact JSON.
    data_dir = (
        params.data_dir
        if (out_dir_prefix is None or not synthetic_data)
        else os.path.join(out_dir_prefix, params.data_dir.split("cheetah_videos")[1])
    )
    grf_df = pd.read_hdf(os.path.join(data_dir, "grf", "data_synth.h5" if synthetic_data else "data.h5"))
    with open(
        (
            os.path.join(data_dir, "grf/autogen-contact.json")
            if synthetic_data
            else os.path.join(params.data_dir, "metadata.json")
        ),
        "r",
        encoding="utf-8",
    ) as f:
        contact_json = json.load(f)
    start_frame = contact_json["start_frame"]
    foot_contact_order = contact_json["contacts"]
    gt_grf_z = {}
    gt_grf_xy = {}
    nfe = params.total_length
    for foot in pe.foot.feet(robot):
        gt_grf_z[foot.name] = [0.0] * nfe
        gt_grf_xy[foot.name] = [[0.0] * foot.nsides for _ in range(nfe)]
        # Check if the foot is in contact sequence.
        if foot.name in foot_contact_order and foot_contact_order[foot.name] is not None:
            grf = grf_df.query(f"force_plate == {foot_contact_order[foot.name][0][2]-1}")
            # Resample data from 3500Hz to 200Hz i.e 2/35 factor.
            if synthetic_data or (not params.kinetic_dataset):
                Fz = grf["Fz"].values
                Fx = grf["Fx"].values
                Fy = grf["Fy"].values
            else:
                Fz = (
                    signal.resample_poly(remove_dc_offset(grf["Fz"].values, 500), up=2, down=35, axis=0)
                    * scale_forces_by
                )
                Fx = (
                    direction
                    * signal.resample_poly(remove_dc_offset(grf["Fx"].values, 500), up=2, down=35, axis=0)
                    * scale_forces_by
                )
                Fy = (
                    direction
                    * signal.resample_poly(remove_dc_offset(grf["Fy"].values, 500), up=2, down=35, axis=0)
                    * scale_forces_by
                )
            foot_on_ground_indices = list(
                range(foot_contact_order[foot.name][0][0], foot_contact_order[foot.name][0][1] + 1)
            )
            for fe in range(1, nfe):
                if (start_frame + fe - 1) in foot_on_ground_indices:
                    if synthetic_data or (not params.kinetic_dataset):
                        z_comp = Fz[fe - 1]
                        x_comp = Fx[fe - 1]
                        y_comp = Fy[fe - 1]
                    else:
                        z_comp = Fz[start_frame + fe - 1]
                        x_comp = Fx[start_frame + fe - 1]
                        y_comp = Fy[start_frame + fe - 1]
                    gt_grf_z[foot.name][fe - 1] = z_comp
                    # Note, range(4) corresponds with the friction polygon estimate using 4 sides. There is also an option to use 8 sides.
                    comps = []
                    for i in range(foot.nsides):
                        # Project force data onto each component of the polygon friction.
                        force_comp = np.array([x_comp, y_comp, 0]).dot(foot.D[i, :])
                        comps.append(force_comp)
                    max_idx = np.argmax(comps)
                    if comps[max_idx] > 0:
                        # There is a force component in the direction foot.D.
                        gt_grf_xy[foot.name][fe - 1][max_idx] = comps[max_idx]

    return gt_grf_z, gt_grf_xy


def init_grf_data(
    robot: pe.system.System3D,
    data_dir: str,
    scale_forces_by: float,
    fix: bool = False,
    init_forces: bool = True,
    synthetic_data: bool = False,
) -> Tuple[Dict, Dict, Dict]:
    grf_df = pd.read_hdf(os.path.join(data_dir, "grf", "data_synth.h5" if synthetic_data else "data.h5"))
    with open(
        os.path.join(data_dir, "grf/autogen-contact.json" if synthetic_data else "metadata.json"),
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)
    start_frame = metadata["start_frame"]
    foot_contact_order = metadata["contacts"]
    # The globally referened data is stored as force plate "8".
    # grf = grf_df.query("force_plate == 8")
    # Resample data from 3500Hz to 200Hz i.e 2/35 factor.
    # Fz = signal.resample_poly(remove_dc_offset(grf["Fz"].values, 500), up=2, down=35, axis=0) * scale_forces_by
    # Fx = signal.resample_poly(remove_dc_offset(grf["Fx"].values, 500), up=2, down=35, axis=0) * scale_forces_by
    # Fy = signal.resample_poly(remove_dc_offset(grf["Fy"].values, 500), up=2, down=35, axis=0) * scale_forces_by
    gt_grf = {}
    horizontal_force_magnitude = {}
    horizontal_force_magnitude_est = {}
    nfe = len(cast(Any, robot.m).fe)
    for foot in pe.foot.feet(robot):
        grfz = foot["GRFz"]
        grfxy = foot["GRFxy"]
        foot_height = foot["foot_height"]
        # Initialise to 0.
        for fe, cp in robot.indices(one_based=True):
            grfz[fe, cp].value = 0.0 if init_forces else np.random.uniform(0.0, 0.05)  # type: ignore
            grfz[fe, cp].fixed = fix
            for i in range(foot.nsides):
                grfxy[fe, cp, i].value = 0.0 if init_forces else np.random.uniform(0.0, 0.05)  # type: ignore
                grfxy[fe, cp, i].fixed = fix
        gt_grf[foot.name] = [0.0] * nfe
        horizontal_force_magnitude[foot.name] = [0.0] * nfe
        horizontal_force_magnitude_est[foot.name] = [0.0] * nfe
        # Check if the foot is in contact sequence.
        if foot.name in foot_contact_order and foot_contact_order[foot.name] is not None:
            grf = grf_df.query(f"force_plate == {foot_contact_order[foot.name][0][2]-1}")
            # Resample data from 3500Hz to 200Hz i.e 2/35 factor.
            if synthetic_data:
                Fz = grf["Fz"].values
                Fx = grf["Fx"].values
                Fy = grf["Fy"].values
            else:
                Fz = (
                    signal.resample_poly(remove_dc_offset(grf["Fz"].values, 500), up=2, down=35, axis=0)
                    * scale_forces_by
                )
                Fx = (
                    -1.0
                    * signal.resample_poly(remove_dc_offset(grf["Fx"].values, 500), up=2, down=35, axis=0)
                    * scale_forces_by
                )
                Fy = (
                    -1.0
                    * signal.resample_poly(remove_dc_offset(grf["Fy"].values, 500), up=2, down=35, axis=0)
                    * scale_forces_by
                )
            foot_on_ground_indices = list(
                range(foot_contact_order[foot.name][0][0], foot_contact_order[foot.name][0][1] + 1)
            )
            for fe, cp in robot.indices(one_based=True):
                if (start_frame + fe - 1) in foot_on_ground_indices:
                    if synthetic_data:
                        z_comp = Fz[fe - 1]
                        x_comp = Fx[fe - 1]
                        y_comp = Fy[fe - 1]
                    else:
                        z_comp = Fz[start_frame + fe - 1]
                        x_comp = Fx[start_frame + fe - 1]
                        y_comp = Fy[start_frame + fe - 1]
                    if init_forces:
                        if z_comp > 0:
                            # foot_height[fe, cp].fix(0)
                            grfz[fe, cp].value = z_comp
                        else:
                            print(f"Warning: z component is not positive at fe {(start_frame + fe - 1)}: {z_comp}")
                    # else:
                    # foot_height[fe, cp].fix(0)
                    # Allow the force to act prior to the contact so we unfix the first foot contact to allow it to be just off the ground before impact.
                    # if (start_frame + fe - 1) == foot_on_ground_indices[0]:
                    # foot_height[fe, cp].fixed = False
                    gt_grf[foot.name][fe - 1] = z_comp
                    # Note, range(4) corresponds with the friction polygon estimate using 4 sides. There is also an option to use 8 sides.
                    comps = []
                    for i in range(foot.nsides):
                        # Project force data onto each component of the polygon friction.
                        force_comp = np.array([x_comp, y_comp, 0]).dot(foot.D[i, :])
                        comps.append(force_comp)
                    max_idx = np.argmax(comps)
                    if init_forces and comps[max_idx] > 0:
                        # There is a force component in the direction foot.D.
                        grfxy[fe, cp, max_idx].value = comps[max_idx]
                    # Check.
                    horizontal_force_magnitude[foot.name][fe - 1] = np.sqrt(x_comp**2 + y_comp**2)
                    horizontal_force_magnitude_est[foot.name][fe - 1] = pyo.value(sum(grfxy[fe, :, :]))
            # Ensure force acts prior to contact, so duplicate the initial force and act it across two timesteps.
            # init_z = Fz[foot_on_ground_indices[0] - start_frame] if synthetic_data else Fz[foot_on_ground_indices[0]]
            # if init_forces:
            # grfz[foot_on_ground_indices[0], :].value = gt_grf[foot.name][np.nonzero(gt_grf[foot.name])[0][0]]

    return gt_grf, horizontal_force_magnitude, horizontal_force_magnitude_est


def prescribe_contact_order(
    robot: pe.system.System3D,
    ground_timings: List[List[int]],
    foot_height_uncertainty: Optional[float] = 0.05,
    min_GRFz: float = 0.01,
) -> None:
    def foot_fix_util(foot, stance_indices: List[int]):
        m = cast(pyo.ConcreteModel, robot.m)
        nfe = len(m.fe)
        flight_indices = list(set(range(1, nfe + 1)) - set(stance_indices))
        GRFz = foot["GRFz"]
        GRFxy = foot["GRFxy"]
        foot_height = foot["foot_height"]
        # phase: flight
        for fe in flight_indices:
            for cp in m.cp:
                GRFz[fe, cp].fix(0)
                GRFxy[fe, cp, :].fix(0)
        # phase: stance
        for fe in stance_indices:
            for cp in m.cp:
                if foot_height_uncertainty is not None:
                    foot_height[fe, cp].setlb(-foot_height_uncertainty)
                    foot_height[fe, cp].setub(foot_height_uncertainty)
                GRFz[fe, cp].setlb(min_GRFz)

    for foot, stance_indices in zip(pe.foot.feet(robot), ground_timings):
        foot_fix_util(foot, stance_indices)


def traj_smoothness(X: np.ndarray, Y: np.ndarray) -> float:
    X = np.asarray(X)
    Y = np.asarray(Y)
    dx = np.linalg.norm(np.diff(X, axis=0), axis=2)
    dy = np.linalg.norm(np.diff(Y, axis=0), axis=2)

    return np.mean(np.abs(dx - dy))


def traj_error(
    X: np.ndarray, Y: np.ndarray, model_name: str = "single view", centered: bool = False
) -> Tuple[pd.DataFrame, np.ndarray, float]:
    # Warning: this function modifies the X and Y inputs so make sure to use a copy of the data if the original is used elsewhere.
    smoothness_error_mm = traj_smoothness(X, Y) * 1000.0
    X = np.asarray(X)
    Y = np.asarray(Y)
    markers = get_markers()
    if centered:
        X -= np.expand_dims(np.mean(X, axis=1), axis=1)
        Y -= np.expand_dims(np.mean(Y, axis=1), axis=1)
    distances = np.sqrt(np.sum((X - Y) ** 2, axis=2))
    trajectory_error_mm = np.mean(distances, axis=1) * 1000.0
    mpjpe_mm = np.mean(distances, axis=0) * 1000.0
    result = pd.DataFrame(mpjpe_mm.reshape(1, len(markers)), columns=markers)
    print(f"{model_name} {'mpjpe' if centered else 'mpe'} [mm]: {float(result.mean(axis=1)):.1f}")
    print(f"{model_name} smoothness error [mm]: {smoothness_error_mm:.1f}")
    result = result.transpose()
    result.columns = ["mpjpe (mm)"]

    return result.astype(float), trajectory_error_mm, smoothness_error_mm


def compare_traj_error(
    params: TrajectoryParams,
    scene: Scene,
    include_kinetic=False,
    kinetic_out_fname: str = "fte",
    out_dir_prefix: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt

    # Gather data for the mult-view, single-view and single view with contraints.
    data_dir = (
        params.data_dir
        if out_dir_prefix is None
        else os.path.join(out_dir_prefix, params.data_dir.split("cheetah_videos")[1][1:])
    )
    fte_multi_view = os.path.join(data_dir, "fte_kinematic", "fte.pickle")
    fte_orig = os.path.join(data_dir, f"fte_kinematic_orig_{scene.cam_idx}", "fte.pickle")
    fte_kinematic = os.path.join(data_dir, f"fte_kinematic_{scene.cam_idx}", "fte.pickle")
    if include_kinetic:
        fte_kinetic = os.path.join(data_dir, f"fte_kinetic_{scene.cam_idx}", f"{kinetic_out_fname}.pickle")
    multi_view_data = data_ops.load_pickle(fte_multi_view)
    single_view_data = data_ops.load_pickle(fte_orig)
    pose_model_data = data_ops.load_pickle(fte_kinematic)
    if include_kinetic:
        kinetic_model_data = data_ops.load_pickle(fte_kinetic)
    single_view_mpjpe_mm, single_view_error, _ = traj_error(
        multi_view_data["positions"].copy(), single_view_data["positions"].copy()
    )
    pose_model_mpjpe_mm, pose_model_error, _ = traj_error(
        multi_view_data["positions"].copy(), pose_model_data["positions"].copy(), "data-driven model"
    )
    _, _, _ = traj_error(
        multi_view_data["positions"].copy(), pose_model_data["positions"].copy(), "data-driven model", True
    )
    if include_kinetic:
        kinetic_model_mpjpe_mm, kinetic_model_error, _ = traj_error(
            multi_view_data["positions"].copy(), kinetic_model_data["positions"].copy(), "physics-based model"
        )
        _, _, _ = traj_error(
            multi_view_data["positions"].copy(), kinetic_model_data["positions"].copy(), "physics-based model", True
        )

    # Plot the error between the single view and snigle view with constraints.
    fig = plt.figure(figsize=(16, 12), dpi=120)
    plt.plot(single_view_error, color=plot_color["gray"], label="Single view")
    # plt.plot([np.mean(single_view_error)] * len(single_view_error), label="mean single-view", linestyle="--")
    plt.plot(pose_model_error, color=plot_color["red"], label="Data-driven")
    # plt.plot([np.mean(pose_model_error)] * len(pose_model_error), label="mean data-driven model", linestyle="--")
    if include_kinetic:
        plt.plot(kinetic_model_error, color=plot_color["orange"], label="Physics-based")
        # plt.plot([np.mean(kinetic_model_error)] * len(kinetic_model_error),
        #          label="mean physics-based model",
        #          linestyle="--")
    plt.xlabel("Finite Element")
    plt.ylabel("Error (mm)")
    ax = plt.gca()
    ax.legend()
    fig.savefig(
        os.path.join(
            os.path.dirname(fte_kinetic if include_kinetic else fte_kinematic),
            "traj_error.pdf" if kinetic_out_fname == "fte" else f"traj_error{kinetic_out_fname[-1]}.pdf",
        )
    )
    plt.cla()

    # Plot the X and Y positions of the single view with constraints for the trajectory.
    # pts_3d_spine_multi = np.asarray(multi_view_data["positions"])[:, 4, :]
    # pts_3d_spine_orig = np.asarray(single_view_data["positions"])[:, 4, :]
    # pts_3d_spine = np.asarray(pose_model_data["positions"])[:, 4, :]
    # # fig = plt.figure()
    # plt.plot(pts_3d_spine_multi[:, 0], pts_3d_spine_multi[:, 1], label="multi-view")
    # plt.plot(pts_3d_spine_orig[:, 0], pts_3d_spine_orig[:, 1], label="single-view")
    # plt.plot(pts_3d_spine[:, 0], pts_3d_spine[:, 1], label="single-view pose prior")
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.ylim((0, 12))
    # ax = plt.gca()
    # ax.legend()
    # fig.savefig(os.path.join(os.path.dirname(fte_kinetic), "x_y_positions.png"))
    # plt.cla()

    # x = np.arange(len(labels))  # the label locations
    # width = 0.35  # the width of the bars
    if include_kinetic:
        df = pd.DataFrame(
            {
                "Single view": single_view_mpjpe_mm["mpjpe (mm)"].to_list(),
                "Data-driven": pose_model_mpjpe_mm["mpjpe (mm)"].to_list(),
                "Physics-based": kinetic_model_mpjpe_mm["mpjpe (mm)"].to_list(),
            },
            index=single_view_mpjpe_mm["mpjpe (mm)"].index,
        )
    else:
        df = pd.DataFrame(
            {
                "Single view": single_view_mpjpe_mm["mpjpe (mm)"].to_list(),
                "Data-driven": pose_model_mpjpe_mm["mpjpe (mm)"].to_list(),
            },
            index=single_view_mpjpe_mm["mpjpe (mm)"].index,
        )
    ax = df.plot(kind="barh", color=[plot_color["gray"], plot_color["red"], plot_color["orange"]])
    fig = ax.get_figure()
    fig.set_size_inches(16, 12)
    fig.set_dpi(120)
    plt.xlabel("Error (mm)")
    plt.ylabel("Joint")
    fig.savefig(
        os.path.join(
            os.path.dirname(fte_kinetic if include_kinetic else fte_kinematic),
            "mpjpe_dist.pdf" if kinetic_out_fname == "fte" else f"mpjpe_dist{kinetic_out_fname[-1]}.pdf",
        )
    )
    plt.cla()

    # Plot the hip and shoulder states to get an idea of how well the gait is tracked.
    # fig, axs = plt.subplots(2, 2, figsize=(16, 12), dpi=120)
    # axs = axs.flatten()
    # p_idx = get_pose_params()
    # x_gt = multi_view_data["x"]
    # x = pose_model_data["x"]
    # states = ("theta_6", "theta_8", "theta_10", "theta_12")
    # titles = ("Left Shoulder Angle", "Right Shoulder Angle", "Left Hip Angle", "Right Hip Angle")
    # for i, state in enumerate(states):
    #     axs[i].plot(x_gt[:, p_idx[state]], label="GT multi-view")
    #     axs[i].plot(x[:, p_idx[state]], label="single-view")
    #     axs[i].set_title(titles[i])
    #     axs[i].legend()
    # fig.savefig(os.path.join(os.path.dirname(fte), "states_comparison.png"))


def project_points(obj_pts, k, d, r, t):
    obj_pts_reshaped = obj_pts.reshape((-1, 1, 3))
    r_vec = cv.Rodrigues(r)[0]
    pts = cv.projectPoints(obj_pts_reshaped, r_vec, t, k, d)[0].reshape((-1, 2))
    return pts


def project_points_fisheye(obj_pts, k, d, r, t):
    obj_pts_reshaped = obj_pts.reshape((-1, 1, 3))
    r_vec = cv.Rodrigues(r)[0]
    pts = cv.fisheye.projectPoints(obj_pts_reshaped, r_vec, t, k, d)[0].reshape((-1, 2))
    return pts


def save_3d_cheetah_as_2d(
    positions_3d_arr,
    out_dir,
    scene_fpath,
    bodyparts,
    project_func,
    start_frame,
    sync_offset_arr: List[int],
    vid_dir=None,
    save_as_csv=True,
    out_fname=None,
):
    # assert os.path.dirname(os.path.dirname(scene_fpath)) in out_dir, "scene_fpath does not belong to the same parent folder as out_dir"

    if vid_dir:
        video_fpaths = sorted(glob(os.path.join(vid_dir, "cam[1-9].mp4")))
    else:
        video_fpaths = sorted(glob(os.path.join(out_dir, "cam[1-9].mp4")))  # check current dir for videos
        if not video_fpaths:
            video_fpaths = sorted(
                glob(os.path.join(os.path.dirname(out_dir), "cam[1-9].mp4"))
            )  # check parent dir for videos

    if video_fpaths:
        k_arr, d_arr, r_arr, t_arr, cam_res = load_scene(scene_fpath, verbose=False)
        assert len(k_arr) == len(video_fpaths)

        xyz_labels = ["x", "y", "likelihood"]  # same format as DLC
        pdindex = pd.MultiIndex.from_product([bodyparts, xyz_labels], names=["bodyparts", "coords"])

        out_fname = os.path.basename(out_dir) if out_fname is None else out_fname
        fpath = ""
        cam_name = ""
        for i in range(len(video_fpaths)):
            position_3d = np.array(positions_3d_arr[i])
            n_frames = len(position_3d)
            projections = project_func(position_3d, k_arr[i], d_arr[i], r_arr[i], t_arr[i])
            out_of_range_indices = np.where((projections > cam_res) | (projections < [0] * 2))[0]
            projections[out_of_range_indices] = np.nan

            data = np.full(position_3d.shape, np.nan)
            data[:, :, 0:2] = projections.reshape((n_frames, -1, 2))

            cam_name = os.path.splitext(os.path.basename(video_fpaths[i]))[0]
            fpath = os.path.join(out_dir, cam_name + "_" + out_fname + ".h5")

            df = pd.DataFrame(
                data.reshape((n_frames, -1)),
                columns=pdindex,
                index=range(start_frame - sync_offset_arr[i], start_frame + n_frames - sync_offset_arr[i]),
            )
            if save_as_csv:
                df.to_csv(os.path.splitext(fpath)[0] + ".csv")
            df.to_hdf(fpath, f"{out_fname}_df", format="table", mode="w")

        fpath = fpath.replace(cam_name, "cam*")
        print("Saved", fpath)
        if save_as_csv:
            print("Saved", os.path.splitext(fpath)[0] + ".csv")
        print()
    else:
        print("Could not save 3D cheetah to 2D - No videos were found in", out_dir, "or", os.path.dirname(out_dir))


def save_optimised_cheetah(positions, out_fpath, extra_data=None):
    file_data = dict(positions=positions)

    if extra_data is not None:
        assert type(extra_data) is dict
        file_data.update(extra_data)

    with open(out_fpath, "wb") as f:
        pickle.dump(file_data, f)
    print("Saved", out_fpath)


def triangulate_points_single_img(img_pts, dist_to_plane, K, D, R, t) -> np.ndarray:
    pts = img_pts.reshape((-1, 1, 2))
    norm_pts = cv.fisheye.undistortPoints(pts, K, D)
    norm_pts = norm_pts.reshape((-1, 2))
    norm_pts = np.append(norm_pts, [[1] for i in range(len(norm_pts))], 1).T
    X_c = dist_to_plane * norm_pts
    X_w = np.dot(R.T, X_c) - np.dot(R.T, t)
    return X_w


def triangulate_points(img_pts_1, img_pts_2, k1, d1, r1, t1, k2, d2, r2, t2):
    pts_1 = img_pts_1.reshape((-1, 1, 2))
    pts_2 = img_pts_2.reshape((-1, 1, 2))
    pts_1 = cv.undistortPoints(pts_1, k1, d1)
    pts_2 = cv.undistortPoints(pts_2, k2, d2)
    p1 = np.hstack((r1, t1))
    p2 = np.hstack((r2, t2))
    pts_4d = cv.triangulatePoints(p1, p2, pts_1, pts_2)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T
    return pts_3d


def triangulate_points_fisheye(img_pts_1, img_pts_2, k1, d1, r1, t1, k2, d2, r2, t2):
    pts_1 = img_pts_1.reshape((-1, 1, 2))
    pts_2 = img_pts_2.reshape((-1, 1, 2))
    pts_1 = cv.fisheye.undistortPoints(pts_1, k1, d1)
    pts_2 = cv.fisheye.undistortPoints(pts_2, k2, d2)
    p1 = np.hstack((r1, t1))
    p2 = np.hstack((r2, t2))
    pts_4d = cv.triangulatePoints(p1, p2, pts_1, pts_2)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T
    return pts_3d


def get_pairwise_3d_points_from_df(points_2d_df, k_arr, d_arr, r_arr, t_arr, triangulate_func, verbose=True):
    n_cams = len(k_arr)
    camera_pairs = [[i % n_cams, (i + 1) % n_cams] for i in range(n_cams)]
    df_pairs = pd.DataFrame(columns=["x", "y", "z"])
    # get pairwise estimates
    for cam_a, cam_b in camera_pairs:
        d0 = points_2d_df[points_2d_df["camera"] == cam_a]
        d1 = points_2d_df[points_2d_df["camera"] == cam_b]
        intersection_df = d0.merge(d1, how="inner", on=["frame", "marker"], suffixes=("_a", "_b"))
        if intersection_df.shape[0] > 0:
            if verbose:
                print(f"Found {intersection_df.shape[0]} pairwise points between camera {cam_a} and {cam_b}")
            cam_a_points = np.array(intersection_df[["x_a", "y_a"]], dtype=np.float32).reshape((-1, 1, 2))
            cam_b_points = np.array(intersection_df[["x_b", "y_b"]], dtype=np.float32).reshape((-1, 1, 2))
            points_3d = triangulate_func(
                cam_a_points,
                cam_b_points,
                k_arr[cam_a],
                d_arr[cam_a],
                r_arr[cam_a],
                t_arr[cam_a],
                k_arr[cam_b],
                d_arr[cam_b],
                r_arr[cam_b],
                t_arr[cam_b],
            )
            intersection_df["x"] = points_3d[:, 0]
            intersection_df["y"] = points_3d[:, 1]
            intersection_df["z"] = points_3d[:, 2]
            df_pairs = pd.concat([df_pairs, intersection_df], ignore_index=True, join="outer", sort=False)
        else:
            if verbose:
                print(f"No pairwise points between camera {cam_a} and {cam_b}")

    if verbose:
        print()
    points_3d_df = df_pairs[["frame", "marker", "x", "y", "z"]].groupby(["frame", "marker"]).mean().reset_index()
    return points_3d_df


def load_scene(fpath, verbose=True):
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
        cam_res = tuple(data["camera_resolution"])
        k_arr = []
        d_arr = []
        r_arr = []
        t_arr = []
        for c in data["cameras"]:
            k_arr.append(c["k"])
            d_arr.append(c["d"])
            r_arr.append(c["r"])
            t_arr.append(c["t"])
        k_arr = np.array(k_arr, dtype=np.float64)
        d_arr = np.array(d_arr, dtype=np.float64)
        r_arr = np.array(r_arr, dtype=np.float64)
        t_arr = np.array(t_arr, dtype=np.float64)
    if verbose:
        print(f"Loaded extrinsics from {fpath}\n")
    return k_arr, d_arr, r_arr, t_arr, cam_res


def find_scene_file(dir_path, scene_fname=None, verbose=True):
    if scene_fname is None:
        n_cams = len(glob(os.path.join(dir_path, "cam[1-9].mp4")))  # reads up to cam9.mp4 only
        scene_fname = f"{n_cams}_cam_scene_sba.json" if n_cams else "[1-9]_cam_scene*.json"

    if dir_path and dir_path != os.path.sep and dir_path != os.path.join("..", "data"):
        scene_fpath = os.path.join(dir_path, "extrinsic_calib", scene_fname)
        # ignore [1-9]_cam_scene_before_corrections.json unless specified
        scene_files = sorted(
            [
                scene_file
                for scene_file in glob(scene_fpath)
                if ("before_corrections" not in scene_file) or (scene_file == scene_fpath)
            ]
        )

        if scene_files:
            k_arr, d_arr, r_arr, t_arr, cam_res = load_scene(scene_files[-1], verbose)
            scene_fname = os.path.basename(scene_files[-1])
            n_cams = int(scene_fname[0])  # assuming scene_fname is of the form "[1-9]_cam_scene*"
            return k_arr, d_arr, r_arr, t_arr, cam_res, n_cams, scene_files[-1]
        else:
            return find_scene_file(os.path.dirname(dir_path), scene_fname, verbose)

    raise FileNotFoundError(ENOENT, os.strerror(ENOENT), os.path.join("extrinsic_calib", scene_fname))


def load_dlc_points_as_df(dlc_df_fpaths, hand_labeled=False, verbose=True):
    dfs = []
    for path in dlc_df_fpaths:
        dlc_df = pd.read_hdf(path)
        if hand_labeled:
            start_frame = int(dlc_df.index[0][-1][3:6])
            end_frame = int(dlc_df.index[-1][-1][3:6])
            dlc_df = dlc_df.set_index(pd.Index(list(range(start_frame, end_frame + 1))))
        dlc_df = (
            dlc_df.droplevel([0], axis=1)
            .swaplevel(0, 1, axis=1)
            .T.unstack()
            .T.reset_index()
            .rename({"level_0": "frame"}, axis=1)
        )
        dlc_df.columns.name = ""
        dfs.append(dlc_df)
    # create new dataframe
    dlc_df = pd.DataFrame(columns=["frame", "camera", "marker", "x", "y", "likelihood"])
    for i, df in enumerate(dfs):
        df["camera"] = i
        df.rename(columns={"bodyparts": "marker"}, inplace=True)
        dlc_df = pd.concat([dlc_df, df], sort=True, ignore_index=True)

    dlc_df = dlc_df[["frame", "camera", "marker", "x", "y", "likelihood"]]
    if verbose:
        print(f"DLC points dataframe:\n{dlc_df}\n")
    return dlc_df


def generate_link_pos_funcs(pos1, pos2, sp_vars) -> tuple:
    assert pos1.shape == (3, 1)
    assert pos2.shape == (3, 1)
    return pe.utils.lambdify_EOM(pos1, sp_vars, test_func=True), pe.utils.lambdify_EOM(pos2, sp_vars, test_func=True)


def get_pose_state(robot: pe.system.System3D) -> tuple:
    state_vars = [[var for var in robot[link.name].q] for link in robot.links]
    length_vars = [robot[link.name].length_sym for link in robot.links]
    sp_vars = sum(state_vars, []) + length_vars
    # Go through each link and determine the position for each dlc marker.
    p_l_eye = robot["neck"].bottom_I + robot["neck"].Rb_I @ sp.Matrix([0, -0.045, 0])
    p_r_eye = robot["neck"].bottom_I + robot["neck"].Rb_I @ sp.Matrix([0, 0.045, 0])
    p_nose = robot["neck"].bottom_I + robot["neck"].Rb_I @ sp.Matrix([-0.055, 0, -0.055])
    p_l_shoulder = robot["bodyF"].bottom_I + robot["bodyF"].Rb_I @ sp.Matrix([0.06, -0.075, -0.15])
    p_r_shoulder = robot["bodyF"].bottom_I + robot["bodyF"].Rb_I @ sp.Matrix([0.06, 0.075, -0.15])
    p_l_hip = robot["base"].top_I + robot["base"].Rb_I @ sp.Matrix([-0.06, -0.06, -0.1])
    p_r_hip = robot["base"].top_I + robot["base"].Rb_I @ sp.Matrix([-0.06, 0.06, -0.1])
    temp_1 = generate_link_pos_funcs(p_l_eye, p_r_eye, sp_vars)
    temp_2 = generate_link_pos_funcs(p_nose, robot["neck"].top_I, sp_vars)
    p_l_eye = temp_1[0]
    p_r_eye = temp_1[1]
    p_nose = temp_2[0]
    p_neck_base = temp_2[1]
    body_link = generate_link_pos_funcs(robot["base"].top_I, robot["base"].bottom_I, sp_vars)
    p_spine = body_link[1]
    p_tail_base = body_link[0]
    tail_link = generate_link_pos_funcs(robot["tail1"].top_I, robot["tail1"].bottom_I, sp_vars)
    p_tail_mid = tail_link[0]
    p_tail_tip = tail_link[1]

    upper_front_left_link = generate_link_pos_funcs(p_l_shoulder, robot["UFL"].bottom_I, sp_vars)
    p_l_shoulder = upper_front_left_link[0]
    p_l_front_knee = upper_front_left_link[1]
    lower_front_left_link = generate_link_pos_funcs(robot["HFL"].top_I, robot["HFL"].bottom_I, sp_vars)
    p_l_front_ankle = lower_front_left_link[0]
    p_l_front_paw = lower_front_left_link[1]

    upper_front_right_link = generate_link_pos_funcs(p_r_shoulder, robot["UFR"].bottom_I, sp_vars)
    p_r_shoulder = upper_front_right_link[0]
    p_r_front_knee = upper_front_right_link[1]
    lower_front_right_link = generate_link_pos_funcs(robot["HFR"].top_I, robot["HFR"].bottom_I, sp_vars)
    p_r_front_ankle = lower_front_right_link[0]
    p_r_front_paw = lower_front_right_link[1]

    upper_back_left_link = generate_link_pos_funcs(p_l_hip, robot["UBL"].bottom_I, sp_vars)
    p_l_hip = upper_back_left_link[0]
    p_l_back_knee = upper_back_left_link[1]
    lower_back_left_link = generate_link_pos_funcs(robot["HBL"].top_I, robot["HBL"].bottom_I, sp_vars)
    p_l_back_ankle = lower_back_left_link[0]
    p_l_back_paw = lower_back_left_link[1]

    upper_back_right_link = generate_link_pos_funcs(p_r_hip, robot["UBR"].bottom_I, sp_vars)
    p_r_hip = upper_back_right_link[0]
    p_r_back_knee = upper_back_right_link[1]
    lower_back_right_link = generate_link_pos_funcs(robot["HBR"].top_I, robot["HBR"].bottom_I, sp_vars)
    p_r_back_ankle = lower_back_right_link[0]
    p_r_back_paw = lower_back_right_link[1]

    return sp_vars, [
        p_nose,
        p_r_eye,
        p_l_eye,
        p_neck_base,
        p_spine,
        p_tail_base,
        p_tail_mid,
        p_tail_tip,
        p_r_shoulder,
        p_r_front_knee,
        p_r_front_ankle,
        p_r_front_paw,
        p_l_shoulder,
        p_l_front_knee,
        p_l_front_ankle,
        p_l_front_paw,
        p_r_hip,
        p_r_back_knee,
        p_r_back_ankle,
        p_r_back_paw,
        p_l_hip,
        p_l_back_knee,
        p_l_back_ankle,
        p_l_back_paw,
    ]


# ========= PROJECTION FUNCTIONS ========
def pt3d_to_2d_fisheye(x, y, z, K, D, R, t):
    x_2d = x * R[0, 0] + y * R[0, 1] + z * R[0, 2] + t.flatten()[0]
    y_2d = x * R[1, 0] + y * R[1, 1] + z * R[1, 2] + t.flatten()[1]
    z_2d = x * R[2, 0] + y * R[2, 1] + z * R[2, 2] + t.flatten()[2]
    # project onto camera plane
    a = x_2d / z_2d
    b = y_2d / z_2d
    # fisheye params
    r = (a**2 + b**2) ** 0.5
    th = pyo.atan(r)
    # distortion
    th_d = th * (1 + D[0] * th**2 + D[1] * th**4 + D[2] * th**6 + D[3] * th**8)
    x_p = a * th_d / (r + 1e-12)
    y_p = b * th_d / (r + 1e-12)
    u = K[0, 0] * x_p + K[0, 2]
    v = K[1, 1] * y_p + K[1, 2]
    return u, v


def pt3d_to_2d(x, y, z, K, D, R, t):
    x_2d = x * R[0, 0] + y * R[0, 1] + z * R[0, 2] + t.flatten()[0]
    y_2d = x * R[1, 0] + y * R[1, 1] + z * R[1, 2] + t.flatten()[1]
    z_2d = x * R[2, 0] + y * R[2, 1] + z * R[2, 2] + t.flatten()[2]
    # project onto camera plane
    a = x_2d / z_2d
    b = y_2d / z_2d
    r = (a**2 + b**2) ** 0.5
    # distortion
    d = 1 + D[0] * r**2 + D[1] * r**4 + D[2] * r**6
    x_p = a * d
    y_p = b * d
    u = K[0, 0] * x_p + K[0, 2]
    v = K[1, 1] * y_p + K[1, 2]
    return u, v


def get_relative_angle_mask():
    return np.array(
        [
            1,  # x_{base_bodyB}
            1,  # y_{base_bodyB}
            1,  # z_{base_bodyB}
            1,  # \phi_{base_bodyB}
            1,  # \theta_{base_bodyB}
            1,  # \psi_{base_bodyB}
            1,  # \phi_{body_F}
            1,  # \theta_{body_F}
            1,  # \psi_{body_F}
            1,  # \phi_{neck}
            1,  # \theta_{neck}
            1,  # \psi_{neck}
            0,  # \phi_{tail0}
            1,  # \theta_{tail0}
            1,  # \psi_{tail0}
            0,  # \phi_{tail1}
            1,  # \theta_{tail1}
            1,  # \psi_{tail1}
            0,  # \phi_{UFL}
            1,  # \theta_{UFL}
            0,  # \psi_{UFL}
            0,  # \phi_{LFL}
            1,  # \theta_{LFL}
            0,  # \psi_{LFL}
            0,  # \phi_{HFL}
            1,  # \theta_{HFL}
            0,  # \psi_{HFL}
            0,  # \phi_{UFR}
            1,  # \theta_{UFR}
            0,  # \psi_{UFR}
            0,  # \phi_{LFR}
            1,  # \theta_{LFR}
            0,  # \psi_{LFR}
            0,  # \phi_{HFR}
            1,  # \theta_{HFR}
            0,  # \psi_{HFR}
            0,  # \phi_{UBL}
            1,  # \theta_{UBL}
            0,  # \psi_{UBL}
            0,  # \phi_{LBL}
            1,  # \theta_{LBL}
            0,  # \psi_{LBL}
            0,  # \phi_{UBR}
            1,  # \theta_{UBR}
            0,  # \psi_{UBR}
            0,  # \phi_{LBR}
            1,  # \theta_{LBR}
            0,  # \psi_{LBR}
            0,  # \phi_{HBL}
            1,  # \theta_{HBL}
            0,  # \psi_{HBL}
            0,  # \phi_{HBR}
            1,  # \theta_{HBR}
            0,  # \psi_{HBR}
        ]
    ).nonzero()


def get_uncertainty_models(relative_angles: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    # measurement standard deviation
    R = np.array(
        [
            1.2,  # nose
            1.24,  # l_eye
            1.18,  # r_eye
            2.08,  # neck_base
            2.04,  # spine
            2.52,  # tail_base
            2.73,  # tail1
            1.83,  # tail2
            3.47,  # r_shoulder
            2.75,  # r_front_knee
            2.69,  # r_front_ankle
            2.24,  # r_front_paw
            3.4,  # l_shoulder
            2.91,  # l_front_knee
            2.85,  # l_front_ankle
            2.27,  # l_front_paw
            3.26,  # r_hip
            2.76,  # r_back_knee
            2.33,  # r_back_ankle
            2.4,  # r_back_paw
            3.53,  # l_hip
            2.69,  # l_back_knee
            2.49,  # l_back_ankle
            2.34,  # l_back_paw
        ],
        dtype=float,
    )
    R_pw = np.array(
        [
            R,
            [
                2.71,
                3.06,
                2.99,
                4.07,
                5.53,
                4.67,
                6.05,
                5.6,
                5.01,
                5.11,
                5.24,
                4.85,
                5.18,
                5.28,
                5.5,
                4.9,
                4.7,
                4.7,
                5.21,
                5.11,
                5.1,
                5.27,
                5.75,
                5.44,
            ],
            [
                2.8,
                3.24,
                3.42,
                3.8,
                4.4,
                5.43,
                5.22,
                7.29,
                8.19,
                6.5,
                5.9,
                6.18,
                8.83,
                6.52,
                6.22,
                6.34,
                6.8,
                6.12,
                5.37,
                5.98,
                7.83,
                6.44,
                6.1,
                6.38,
            ],
        ],
        dtype=float,
    )
    # Provides some extra uncertainty to the measurements to accomodate for the rigid body body assumption.
    R_pw *= 2

    Q = [
        4,  # x_{base_bodyB}
        7,  # y_{base_bodyB}
        5,  # z_{base_bodyB}
        13,  # \phi_{base_bodyB}
        9,  # \theta_{base_bodyB}
        26,  # \psi_{base_bodyB}
        10,  # \phi_{body_F}
        53,  # \theta_{body_F}
        34,  # \psi_{body_F}
        32,  # \phi_{neck}
        18,  # \theta_{neck}
        12,  # \psi_{neck}
        0,  # \phi_{tail0}
        90,  # \theta_{tail0}
        43,  # \psi_{tail0}
        0,  # \phi_{tail1}
        118,  # \theta_{tail1}
        51,  # \psi_{tail1}
        0,  # \phi_{UFL}
        247,  # \theta_{UFL}
        0,  # \psi_{UFL}
        0,  # \phi_{LFL}
        186,  # \theta_{LFL}
        0,  # \psi_{LFL}
        0,  # \phi_{HFL}
        91,  # \theta_{HFL}
        0,  # \psi_{HFL}
        0,  # \phi_{UFR}
        194,  # \theta_{UFR}
        0,  # \psi_{UFR}
        0,  # \phi_{LFR}
        164,  # \theta_{LFR}
        0,  # \psi_{LFR}
        0,  # \phi_{HFR}
        91,  # \theta_{HFR}
        0,  # \psi_{HFR}
        0,  # \phi_{UBL}
        295,  # \theta_{UBL}
        0,  # \psi_{UBL}
        0,  # \phi_{LBL}
        243,  # \theta_{LBL}
        0,  # \psi_{LBL}
        0,  # \phi_{UBR}
        334,  # \theta_{UBR}
        0,  # \psi_{UBR}
        0,  # \phi_{LBR}
        149,  # \theta_{LBR}
        0,  # \psi_{LBR}
        0,  # \phi_{HBL}
        132,  # \theta_{HBL}
        0,  # \psi_{HBL}
        0,  # \phi_{HBR}
        132,  # \theta_{HBR}
        0,  # \psi_{HBR}
    ]
    Q = np.array(Q, dtype=float) ** 2
    # Q[Q == 0] = 1

    return R_pw, Q[Q != 0] if relative_angles else Q


def get_markers():
    return [
        "nose",
        "r_eye",
        "l_eye",
        "neck_base",
        "spine",
        "tail_base",
        "tail1",
        "tail2",
        "r_shoulder",
        "r_front_knee",
        "r_front_ankle",
        "r_front_paw",
        "l_shoulder",
        "l_front_knee",
        "l_front_ankle",
        "l_front_paw",
        "r_hip",
        "r_back_knee",
        "r_back_ankle",
        "r_back_paw",
        "l_hip",
        "l_back_knee",
        "l_back_ankle",
        "l_back_paw",
    ]


def get_dlc_marker_indices():
    return {
        "nose": 23,
        "r_eye": 0,
        "l_eye": 1,
        "neck_base": 24,
        "spine": 6,
        "tail_base": 22,
        "tail1": 11,
        "tail2": 12,
        "l_shoulder": 13,
        "l_front_knee": 14,
        "l_front_ankle": 15,
        "l_front_paw": 16,
        "r_shoulder": 2,
        "r_front_knee": 3,
        "r_front_ankle": 4,
        "r_front_paw": 5,
        "l_hip": 17,
        "l_back_knee": 18,
        "l_back_ankle": 19,
        "l_back_paw": 20,
        "r_hip": 7,
        "r_back_knee": 8,
        "r_back_ankle": 9,
        "r_back_paw": 10,
    }


def get_pairwise_graph():
    return {
        "r_eye": [23, 1],
        "l_eye": [23, 0],
        "nose": [0, 1],
        "neck_base": [6, 23],
        "spine": [22, 24],
        "tail_base": [6, 11],
        "tail1": [6, 22],
        "tail2": [11, 22],
        "l_shoulder": [14, 24],
        "l_front_knee": [13, 15],
        "l_front_ankle": [13, 14],
        "l_front_paw": [14, 15],
        "r_shoulder": [3, 24],
        "r_front_knee": [2, 4],
        "r_front_ankle": [2, 3],
        "r_front_paw": [3, 4],
        "l_hip": [18, 22],
        "l_back_knee": [17, 19],
        "l_back_ankle": [17, 18],
        "l_back_paw": [18, 19],
        "r_hip": [8, 22],
        "r_back_knee": [7, 9],
        "r_back_ankle": [7, 8],
        "r_back_paw": [8, 9],
    }


def redescending_loss(err, a, b, c) -> float:
    # outlier rejecting cost function
    def func_step(start, x):
        return 1 / (1 + np.e ** (-1 * (x - start)))

    def func_piece(start, end, x):
        return func_step(start, x) - func_step(end, x)

    e = abs(err)
    cost = 0.0
    cost += (1 - func_step(a, e)) / 2 * e**2
    cost += func_piece(a, b, e) * (a * e - (a**2) / 2)
    cost += func_piece(b, c, e) * (a * b - (a**2) / 2 + (a * (c - b) / 2) * (1 - ((c - e) / (c - b)) ** 2))
    cost += func_step(c, e) * (a * b - (a**2) / 2 + (a * (c - b) / 2))
    return cost


def redescending_smooth_loss(r, c, arctan_func) -> float:
    cost = 0.25 * c**2 * (arctan_func(r / c) ** 2 + ((c * r) ** 2) / (c**4 + r**4))
    return cost


def cauchy_loss(r, c, log_func) -> float:
    cost = c**2 * (log_func(1 + (r / c) ** 2))
    return cost


def fair_loss(r, c, log_func) -> float:
    cost = c**2 * ((abs(r) / c) - log_func(1 + (abs(r) / c)))
    return cost


def positive_zero_crossings(x: np.ndarray) -> Tuple[float, List]:
    zero_crossing_count = 0
    arg_zero_crossing = []
    x = x[np.nonzero(x)]
    for i in range(1, len(x)):
        if (x[i - 1]) < 0 and x[i] > 0:
            zero_crossing_count += 1
            arg_zero_crossing.append(i + 2)
            arg_zero_crossing.append(i + 1)
            arg_zero_crossing.append(i)
            arg_zero_crossing.append(i - 1)
            arg_zero_crossing.append(i - 2)

    return zero_crossing_count, arg_zero_crossing


def group_by_consecutive_values(x: Union[List, np.ndarray]):
    spl = [0] + [i for i in range(1, len(x)) if x[i] - x[i - 1] > 1] + [None]
    return [x[b:e] for (b, e) in [(spl[i - 1], spl[i]) for i in range(1, len(spl))]]


def find_minimum_foot_height(x: Union[List, np.ndarray], region: Tuple[int, int]):
    arg_min = np.argmin(x[region[0] : region[1]])
    return region[0] + arg_min
