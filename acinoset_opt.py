import os
from time import time
from types import FunctionType
import dill
import json
from dataclasses import dataclass
import platform
from typing import Iterable, List, Tuple, Dict, Optional, Any, cast
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from pyomo.util.infeasible import log_infeasible_constraints
import shared.physical_education as pe
import cheetah
import acinoset_misc as misc
from acinoset_models import MotionModel, PoseModelGMM
from common.py_utils import data_ops


@dataclass
class CheetahEstimator:
    name: str
    data_path: str
    model: pe.system.System3D
    params: misc.TrajectoryParams
    scene: misc.Scene
    scale_forces_by: float
    position_funcs: List[List[FunctionType]]
    kinematic_model: bool
    enable_eom_slack: bool
    costs: Optional[Dict] = None
    synthesised_grf: Optional[Dict] = None
    com_pos: Optional[np.ndarray] = None
    com_vel: Optional[np.ndarray] = None
    opt_time_s: Optional[float] = None
    pose_model: Optional[PoseModelGMM] = None
    motion_model: Optional[MotionModel] = None

    def reset_pyomo_model(
        self,
        extend_by: Optional[int] = None,
        include_camera_constraints: bool = True,
    ):
        robot = self.model
        params = self.params
        scene = self.scene
        params.end_frame = params.end_frame if extend_by is None else params.end_frame + extend_by
        N = params.end_frame - params.start_frame
        if self.kinematic_model:
            robot.make_pyomo_model(nfe=N,
                                   collocation="implicit_euler",
                                   total_time=N / scene.fps,
                                   include_dynamics=False)
        else:
            robot.make_pyomo_model(nfe=N,
                                   collocation="implicit_euler",
                                   total_time=N / scene.fps,
                                   include_eom_slack=True)
        cheetah.add_pyomo_constraints(robot, kinetic_dataset=params.kinetic_dataset)
        if include_camera_constraints:
            dlc_dir = os.path.join(params.data_dir, "dlc" if not params.hand_labeled_data else "dlc_hand_labeled")
            misc.create_camera_contraints(robot,
                                          params,
                                          scene,
                                          dlc_dir,
                                          self.position_funcs,
                                          params.hand_labeled_data,
                                          kinetic_dataset=params.kinetic_dataset)

    def init_torques(self) -> None:
        robot = self.model
        feet = cast(List[pe.foot.Foot3D], pe.foot.feet(robot))
        motors = cast(List[pe.motor.Motor3D], pe.motor.torques(robot))
        torque_vars = tuple(pe.utils.flatten(torque.input_torques for torque in motors))
        padding = [7, 8, 10, 11, 13, 14, 16, 17, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52]
        Fr = pe.utils.flatten(link.constraint_forces for link in robot.links)
        len_vars = []
        mass_vars = []
        radius_vars = []
        for link in robot.links:
            len_vars.append((link.length_sym, link.length))
            mass_vars.append((link.mass_sym, link.mass))
            radius_vars.append((link.radius_sym, link.radius))
        F_joint = list(zip(Fr, [0] * len(Fr)))
        to_sub = F_joint + len_vars + mass_vars + radius_vars
        static_eom = robot.eom.xreplace({key: value for (key, value) in to_sub})
        for fe in cast(Any, robot.m).fe:
            q_vars = []
            dq_vars = []
            ddq_vars = []
            for link in robot.links:
                len_vars.append((link.length_sym, link.length))
                mass_vars.append((link.mass_sym, link.mass))
                radius_vars.append((link.radius_sym, link.radius))
                for i, q in enumerate(link.pyomo_sets["q_set"]):
                    q_vars.append((link.q[i], link["q"][fe, 1, q].value))
                    dq_vars.append((link.dq[i], link["dq"][fe, 1, q].value))
                    ddq_vars.append((link.ddq[i], link["ddq"][fe, 1, q].value))
            F_lambda = []
            for f in feet:
                Lz = f.Lz
                fric_set = f.pyomo_sets["fric_set"]
                Lx = f.Lx
                for fric in fric_set:
                    F_lambda.append((Lx[fric], f["GRFxy"][fe, 1, fric].value * self.scale_forces_by))
                F_lambda.append((Lz, f["GRFz"][fe, 1].value * self.scale_forces_by))
            to_sub = F_lambda + q_vars + dq_vars + ddq_vars
            torque_eom = static_eom.xreplace({key: value for (key, value) in to_sub})
            output = sp.linsolve(pe.utils.flatten(torque_eom[padding, :].tolist()), torque_vars)
            output = dict(zip(torque_vars, output.args[0]))
            for m in motors:
                for idx in m.pyomo_sets["Tc_set"]:
                    try:
                        m.pyomo_vars["Tc"][fe, idx].value = float(output[m.input_torques[idx]]) / self.scale_forces_by
                    except TypeError:
                        m.pyomo_vars["Tc"][fe, idx].value = m.pyomo_vars["Tc"][fe - 1, idx].value if fe > 1 else 0.0
                        continue

    def calc_grf_eom(self) -> None:
        robot = self.model
        simp_func = lambda x: pe.utils.parsimp(x, nprocs=1)
        q, dq, ddq = robot.get_state_vars()
        Ek, Ep, _, _ = pe.utils.calc_velocities_and_energies(pe.system.getattrs(robot.links, 'Pb_I'),
                                                             pe.system.getattrs(robot.links, 'Rb_I'),
                                                             pe.system.getattrs(robot.links, 'mass_sym'),
                                                             pe.system.getattrs(robot.links, 'inertia'),
                                                             q,
                                                             dq,
                                                             g=9.81)
        Ek = simp_func(Ek)
        Ep = simp_func(Ep)
        M, C, G = pe.utils.manipulator_equation(Ek, Ep, q, dq)
        M = simp_func(M)
        C = simp_func(C)
        G = simp_func(G)
        feet = pe.foot.feet(robot)
        Q = sp.zeros(*q.shape)
        for f in feet:
            Q += f.calc_eom(q, dq, ddq)
        B = simp_func(Q)
        force_scale = sp.Symbol('F_{scale}')
        to_sub = {
            force: force * force_scale
            for force in [
                *pe.utils.flatten(foot.Lx for foot in feet),
                *[foot.Lz for foot in feet],
            ]
        }
        total_mass = sum(link.mass for link in robot.links)
        grf_eom = M[:6, :] @ ddq + C[:6, :] + G[:6, :] - B[:6, :]
        eom = simp_func(grf_eom.xreplace(to_sub))
        len_vars = []
        mass_vars = []
        radius_vars = []
        for link in robot.links:
            len_vars.append((link.length_sym, link.length))
            mass_vars.append((link.mass_sym, link.mass))
            radius_vars.append((link.radius_sym, link.radius))
        to_sub = len_vars + mass_vars + radius_vars + [(force_scale, total_mass * 9.81)]
        eom = eom.xreplace({key: value for (key, value) in to_sub})
        pe.visual.info(f'Number of operations in EOM is {sp.count_ops(eom)}')
        # the lambdifying step actually takes quite long
        from pyomo.environ import atan
        func_map = {
            'sqrt': lambda x: (x + 1e-9)**(1 / 2),
            'atan': atan,
            'atan2': lambda y, x: 2 * atan(y / ((x**2 + y**2 + 1e-9)**(1 / 2) + x))
        }
        eom_vars = pe.utils.flatten(q.tolist()) + pe.utils.flatten(dq.tolist()) + pe.utils.flatten(
            ddq.tolist()) + [foot.Lz for foot in feet] + pe.utils.flatten(foot.Lx for foot in feet)
        grf_eom_f = pe.utils.lambdify_EOM(eom, eom_vars, func_map=func_map)
        with open(f"./models/{self.name}_grf_eom", "wb") as f:
            dill.dump(grf_eom_f, f)

    def estimate_grf(self, plot: bool = False, monocular: bool = False, out_dir_prefix: Optional[str] = None):
        eom = None
        with open(f"./models/{self.name}_grf_eom", "rb") as f:
            eom = dill.load(f)

        m = pyo.ConcreteModel(name="GRF")
        m.q_set = pyo.Set(initialize=range(54), name='q_set', ordered=True)
        m.fric_set = pyo.Set(initialize=range(4), name='fric_set', ordered=True)
        m.feet = pyo.Set(initialize=[foot.name for foot in pe.foot.feet(self.model)], name='feet_set')
        m.GRFxy = pyo.Var(m.feet, m.fric_set, name='GRFxy', bounds=(0, 5.0))
        m.GRFz = pyo.Var(m.feet, name='GRFz', bounds=(0, 5.0))

        # Determine which feet in contact.
        def def_friction_polyhedron(m, f):
            return 1.3 * m.GRFz[f] >= sum(m.GRFxy[f, :])

        m.friction_polyhedron_constr = pyo.Constraint(m.feet, rule=def_friction_polyhedron)
        # Init from previous run FTE.
        data_dir = self.params.data_dir if out_dir_prefix is None else os.path.join(
            out_dir_prefix,
            self.params.data_dir.split("cheetah_videos")[1][1:])
        # fte_states = data_ops.load_pickle(os.path.join(self.params.data_dir, "dynamic_auto", "fte.pickle"))
        fte_states = data_ops.load_pickle(
            os.path.join(data_dir, "fte_kinematic" if not monocular else f"fte_kinematic_{self.scene.cam_idx}",
                         "fte.pickle"))
        with open(os.path.join(data_dir, "grf/autogen-contact.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
        start_frame = metadata["start_frame"]
        end_frame = metadata["end_frame"]
        foot_contact_order = metadata["contacts"]
        N = end_frame - start_frame
        grfz_est = {k: [] for k in m.feet}
        grfxy_est = {k: [] for k in m.feet}
        # grfz_gt, grfxy_gt = misc.get_grf_profile(self.model, self.params, -1.0, 1 / self.scale_forces_by, False)
        grfz_gt = []
        for fe in range(N):
            init_q = fte_states["q"][fe, :]
            init_dq = fte_states["dq"][fe, :]
            init_ddq = fte_states["ddq"][fe, :]
            pe.utils.remove_constraint_if_exists(m, "q")
            pe.utils.remove_constraint_if_exists(m, "dq")
            pe.utils.remove_constraint_if_exists(m, "ddq")
            m.q = pyo.Param(m.q_set, name='q', initialize=lambda m, i: init_q[i])
            m.dq = pyo.Param(m.q_set, name='dq', initialize=lambda m, i: init_dq[i])
            m.ddq = pyo.Param(m.q_set, name='ddq', initialize=lambda m, i: init_ddq[i])
            is_contact = False
            for f in m.feet:
                m.GRFz[f].fix(0)
                m.GRFxy[f, :].fix(0)
                if f in foot_contact_order and foot_contact_order[f] is not None:
                    foot_on_ground_indices = []
                    for contact_seq in foot_contact_order[f]:
                        foot_on_ground_indices = foot_on_ground_indices + list(range(contact_seq[0], contact_seq[1]))
                    if (start_frame + fe) in foot_on_ground_indices:
                        m.GRFz[f].fixed = False
                        m.GRFxy[f, :].fixed = False
                        is_contact = True
            if not is_contact:
                # No contact.
                for f in m.feet:
                    grfz_est[f].append(0)
                    grfxy_est[f].append([0 for i in m.fric_set])
                continue
            eom_error = 0
            eom_vars = [*m.q[:], *m.dq[:], *m.ddq[:], *m.GRFz[:], *m.GRFxy[:, :]]
            for i in range(6):
                eom_error += (eom[i](eom_vars))**2
            pe.utils.remove_constraint_if_exists(m, "cost")
            m.cost = pyo.Objective(expr=pyo.sqrt(eom_error))
            # Solve...
            results = pe.utils.default_solver(
                max_mins=60,
                solver="ma97",
                warm_start_init_point=False,
                OF_hessian_approximation="exact",
            ).solve(m, tee=False)
            for f in m.feet:
                grfz_est[f].append(m.GRFz[f].value if m.GRFz[f].value >= 0 else 0)
                grfxy_est[f].append([m.GRFxy[f, i].value if m.GRFxy[f, i].value >= 0 else 0 for i in m.fric_set])

        if plot:
            _ = plt.figure(figsize=(16, 9), dpi=120)
            fig_idx = 1
            for foot in grfz_est.keys():
                ax = plt.subplot(2, 2, fig_idx)
                GRFz = grfz_est[foot]
                ax.plot(GRFz, label="Estimate")
                # ax.plot(grfz_gt[foot], label="GT")
                ax.set_title(foot)
                ax.legend()
                fig_idx += 1
            plt.show(block=False)
            plt.close()

        return grfz_est, grfxy_est

    def load(self, fte_name: str):
        self.model.init_from_file(os.path.join(self.params.data_dir, fte_name, "cheetah.pickle"),
                                  skip_if_fixed=False,
                                  skip_if_not_None=False,
                                  fix=False)

    def save(self, out_dir: str, fname: str = "fte", out_dir_prefix: Optional[str] = None):
        # Save trajectory.
        robot, m = self.model, cast(pyo.ConcreteModel, self.model.m)
        params = self.params
        scene = self.scene
        markers = misc.get_markers()
        if out_dir_prefix:
            out_dir = os.path.join(out_dir_prefix, self.data_path, "fte" if out_dir is None else out_dir)
        else:
            out_dir = os.path.join(params.data_dir, "fte" if out_dir is None else out_dir)
        os.makedirs(out_dir, exist_ok=True)
        q_optimised = pe.utils.get_vals(robot["base"].pyomo_vars["q"],
                                        (robot["base"].pyomo_sets["q_set"], )).squeeze(axis=1)
        dq_optimised = pe.utils.get_vals(robot["base"].pyomo_vars["dq"],
                                         (robot["base"].pyomo_sets["q_set"], )).squeeze(axis=1)
        ddq_optimised = pe.utils.get_vals(robot["base"].pyomo_vars["ddq"],
                                          (robot["base"].pyomo_sets["q_set"], )).squeeze(axis=1)
        link_lengths = [robot["base"].length]
        for link in robot.links[1:]:
            q_optimised = np.concatenate(
                (q_optimised, pe.utils.get_vals(link.pyomo_vars["q"], (link.pyomo_sets["q_set"], )).squeeze(axis=1)),
                axis=1)
            dq_optimised = np.concatenate(
                (dq_optimised, pe.utils.get_vals(link.pyomo_vars["dq"], (link.pyomo_sets["q_set"], )).squeeze(axis=1)),
                axis=1)
            ddq_optimised = np.concatenate(
                (ddq_optimised, pe.utils.get_vals(link.pyomo_vars["ddq"],
                                                  (link.pyomo_sets["q_set"], )).squeeze(axis=1)),
                axis=1)
            link_lengths.append(link.length)
        link_lengths = np.tile(link_lengths, q_optimised.shape[0]).reshape(q_optimised.shape[0], -1)
        states = np.concatenate((q_optimised, link_lengths), axis=1)
        # Save estimated torque values.
        nfe = len(cast(Any, m.fe))
        torques = {}
        for motor in pe.motor.torques(robot):
            torques[motor.name] = np.array([
                tau.value for tau in
                [motor.pyomo_vars["Tc"][fe, idx] for fe in cast(Iterable, m.fe) for idx in motor.pyomo_sets["Tc_set"]]
            ]).reshape((nfe, -1))
        meas_err = pe.utils.get_vals(m.slack_meas, (m.C, m.L, m.D2, m.W))
        x, dx, ddx = [], [], []
        for fe, _ in robot.indices(one_based=True):
            x.append(
                np.array(pe.utils.flatten(misc.get_relative_angles(q_optimised,
                                                                   fe - 1)))[misc.get_relative_angle_mask()])
            dx.append(
                np.array(pe.utils.flatten(misc.get_relative_angles(dq_optimised,
                                                                   fe - 1)))[misc.get_relative_angle_mask()])
            ddx.append(
                np.array(pe.utils.flatten(misc.get_relative_angles(ddq_optimised,
                                                                   fe - 1)))[misc.get_relative_angle_mask()])
        output = dict(x=np.array(x),
                      dx=np.array(dx),
                      ddx=np.array(ddx),
                      q=q_optimised,
                      dq=dq_optimised,
                      ddq=ddq_optimised,
                      com_pos=self.com_pos,
                      com_vel=self.com_vel,
                      tau=torques,
                      meas_err=meas_err,
                      obj_cost=self.get_objective_cost(),
                      processing_time_s=self.opt_time_s)
        positions_arr = []
        for c in range(1, scene.n_cams + 1):
            tau = m.shutter_delay[c].value if hasattr(m, "shutter_delay") else 0.0
            positions_arr.append(
                np.array([[[
                    self.position_funcs[l][0](states[fe - 1, :]) + robot["base"]["dq"][fe, 1, "x"].value * tau +
                    robot["base"]["ddq"][fe, 1, "x"].value * (tau**2), self.position_funcs[l][1](states[fe - 1, :]) +
                    robot["base"]["dq"][fe, 1, "y"].value * tau + robot["base"]["ddq"][fe, 1, "y"].value * (tau**2),
                    self.position_funcs[l][2](states[fe - 1, :]) + robot["base"]["dq"][fe, 1, "z"].value * tau +
                    robot["base"]["ddq"][fe, 1, "z"].value * (tau**2)
                ] for l in range(len(markers))] for fe in m.fe]))
        out_fpath = os.path.join(out_dir, f"{fname}.pickle")
        # Extract the frame offsets to synchronise all cameras.
        sync_offset_arr = [0] * scene.n_cams
        if params.sync_offset is not None:
            for offset in params.sync_offset:
                sync_offset_arr[offset["cam"]] = offset["frame"]
        misc.save_optimised_cheetah(positions_arr[0],
                                    out_fpath,
                                    extra_data=dict(**output, start_frame=params.start_frame))
        misc.save_3d_cheetah_as_2d(positions_arr,
                                   out_dir,
                                   scene.scene_fpath,
                                   markers,
                                   misc.project_points if params.kinetic_dataset else misc.project_points_fisheye,
                                   params.start_frame,
                                   sync_offset_arr,
                                   vid_dir=params.data_dir,
                                   out_fname=fname)
        # Save robot.
        if not self.kinematic_model:
            robot.save_data_to_file(os.path.join(out_dir, "cheetah.pickle"), "Auto Save", True)

    def joint_error(self, q: np.ndarray) -> Tuple[float, float]:
        robot = self.model
        q_optimised = pe.utils.get_vals(robot["base"].pyomo_vars["q"],
                                        (robot["base"].pyomo_sets["q_set"], )).squeeze(axis=1)
        base_error = misc.rmse(q[:, :6], q_optimised)
        for link in robot.links[1:]:
            q_optimised = np.concatenate(
                (q_optimised, pe.utils.get_vals(link.pyomo_vars["q"], (link.pyomo_sets["q_set"], )).squeeze(axis=1)),
                axis=1)
        relative_error = misc.rmse(q[:, 6:], q_optimised[:, 6:])

        return base_error, relative_error

    def solution_details(self, print_infeasible_contraints: bool = False, tolerance: float = 1e-1):
        m = cast(pyo.ConcreteModel, self.model.m)
        print("Total cost:", pyo.value(m.cost))
        if self.costs:
            for k, v in self.costs.items():
                print(f"-- {k}: {pyo.value(v)}")
        if print_infeasible_contraints:
            print("Infeasible constraints:")
            log_infeasible_constraints(self.model.m, tol=tolerance, log_expression=True, log_variables=True)
        if hasattr(m, "shutter_delay"):
            print("Shutter delay estimation:", [m.shutter_delay[c].value for c in m.C])

    def is_solution_acceptable(self, results: Any, base_error: float, relative_error: float):
        if not (results.solver.status == pyo.SolverStatus.ok
                and results.solver.termination_condition == pyo.TerminationCondition.optimal):
            return False
        # if (base_error > 0.20 or relative_error > 0.50):
        #     return False
        return True

    def get_objective_cost(self) -> float:
        m = cast(pyo.ConcreteModel, self.model.m)
        return pyo.value(m.cost)


def init_trajectory(root_dir: str,
                    data_path: str,
                    cheetah_name: str,
                    kinetic_dataset: bool,
                    solver_path: str,
                    start_frame: int = -1,
                    end_frame: int = -1,
                    dlc_thresh: float = 0.5,
                    include_camera_constraints: bool = True,
                    enable_eom_slack: bool = True,
                    kinematic_model: bool = False,
                    monocular_enable: bool = False,
                    override_monocular_cam: Optional[int] = None,
                    disable_contact_lcp: bool = True,
                    bound_eom_error: Optional[Tuple[float, float]] = None,
                    shutter_delay_estimation: bool = False,
                    enable_ppm: bool = False,
                    hand_labeled_data: bool = False) -> CheetahEstimator:
    """Initialises a trajectory optimisation problem for the `CheetahEstimator` object.

    Args:
        root_dir (str): The root directory that contains the cheetah videos.
        data_path (str): The data path that is relative to the root directory to select a particular trial, e.g. 2019_03_07/phantom/run.
        cheetah_name (str): The name of the cheetah subject, e.g. phantom. This value is usually present in the `data_path`.
        kinetic_dataset (bool): Flag to determine whether we are using the kinetic dataset or AcinoSet. These are the two datasets supported.
        start_frame (int, optional): Start frame of the video to use in the estimation. Defaults to -1 which instead uses the value inside the `metadata.json`.
        end_frame (int, optional): End frame of the video to use in the estimation. Defaults to -1 which instead uses the value inside the `metadata.json`.
        dlc_thresh (float, optional): The DeepLabCut threshold to discard 2D pose estimates. Defaults to 0.5.
        include_camera_constraints (bool, optional): Flag that controls whether we want the camera constraints setup. Defaults to True.
        enable_eom_slack (bool, optional): Flag that enables a noise variable on equation of motion (EOM), i.e. EOM=σ instead of EOM=0. Defaults to True.
        kinematic_model (bool, optional): Flag that indicates if we are using a kinematic model or kinetic model of the cheetah. Note that certain methods requires one or the other, but this will be communicated when running said method. Defaults to False.
        monocular_enable (bool, optional): Flag that indicates whether we are using monocular or multi-view setup. Defaults to False.
        override_monocular_cam (Optional[int], optional): It is possible to override the set view point for monocular using this variable. Defaults to None which instead uses the value inside the `metadata.json`.
        disable_contact_lcp (bool, optional): Flag disables the use of a linear complementarity constraint to describe the contacts. Defaults to True.
        bound_eom_error (Optional[Tuple[float, float]], optional): Bound the EOM noise variable, i.e. b<=σ<=a. Defaults to None.
        shutter_delay_estimation (bool, optional): Enables the shutter delay estimation for the multi-view camera setup. Defaults to False.
        enable_ppm (bool, optional): Enables the pairwise pseudo measurements (PPM) in the optimisation problem. Defaults to False.
        hand_labeled_data (bool, optional): Flag to use hand labelled data instead of DeepLabCut predictions. Defaults to False.

    Returns:
        CheetahEstimator: cheetah estimator object that stores and manages the data related to the trajectory.
    """
    if cheetah_name not in ("jules", "phantom", "shiraz", "arabia"):
        cheetah_name = "acinoset"
    model_name = f"{cheetah_name}-02" if kinetic_dataset else cheetah_name
    robot_name = f"./models/cheetah-{model_name}-spine-base-kinematics" if kinematic_model else f"./models/cheetah-{model_name}-spine-base-test"
    with open(f"{robot_name}_tmp.robot", "rb") as f:
        robot, _ = dill.load(f)
    ground_plane_height = 0.0
    cam_idx = None
    if start_frame < 0 or end_frame < 0:
        with open(os.path.join(root_dir, data_path, "metadata.json"), "r", encoding="utf-8") as f:
            metadata = json.load(f)
        start_frame = metadata["start_frame"]
        end_frame = metadata["end_frame"]
        sync_offset = metadata["cam_sync"]
        ground_plane_height = metadata["ground_plane_height"]
        cam_idx = metadata["monocular_cam"] if monocular_enable else None
        cam_idx = cam_idx if override_monocular_cam is None else override_monocular_cam
        total_length = metadata["end_frame"] - start_frame
    else:
        total_length = end_frame - start_frame
        sync_offset = None
    N = end_frame - start_frame
    data_dir = os.path.join(root_dir, data_path)
    assert os.path.exists(data_dir)
    dlc_dir = os.path.join(data_dir, "dlc" if not hand_labeled_data else "dlc_hand_labeled")
    assert os.path.exists(dlc_dir)
    # ========= IMPORT CAMERA & SCENE PARAMS ========
    k_arr, d_arr, r_arr, t_arr, cam_res, n_cams, scene_fpath = misc.find_scene_file(data_dir)
    fps = 200
    if not kinetic_dataset and "2019" in data_path:
        fps = 120
    elif not kinetic_dataset and "2017" in data_path:
        fps = 90
    d_arr = d_arr.reshape((-1, 4))
    params = misc.TrajectoryParams(data_dir, start_frame, end_frame, total_length, dlc_thresh, sync_offset,
                                   hand_labeled_data, kinetic_dataset, shutter_delay_estimation, enable_ppm)
    scene = misc.Scene(scene_fpath, k_arr, d_arr, r_arr, t_arr, cam_res, fps, n_cams, cam_idx)
    print("Make Pyomo model for the cheetah")
    # Change bounds on torques and GRF.
    if not kinetic_dataset:
        for foot in pe.foot.feet(robot):
            cast(pe.foot.Foot3D, foot).GRFz_max = 5.0
            cast(pe.foot.Foot3D, foot).GRFxy_max = 5.0
    for motor in pe.motor.torques(robot):
        motor.torque_bounds = (None, None)  # TODO: Should we constrain this to valid regoins?
    pe.foot.Foot3D.ground_plane_height = ground_plane_height
    if disable_contact_lcp:
        # Remove the linear complementarity constraints for a simpler model.
        print("Disable LCP constraints")
        for foot in pe.foot.feet(robot):
            cast(pe.foot.Foot3D, foot).enable_lcp(False)
            cast(pe.foot.Foot3D, foot).friction_coeff = 0.8
    if kinematic_model:
        robot.make_pyomo_model(nfe=N, collocation="implicit_euler", total_time=N / fps, include_dynamics=False)
    else:
        robot.make_pyomo_model(nfe=N,
                               collocation="implicit_euler",
                               total_time=N / fps,
                               include_eom_slack=enable_eom_slack,
                               bound_eom_error=bound_eom_error)
    cheetah.add_pyomo_constraints(robot, kinetic_dataset=kinetic_dataset)
    if include_camera_constraints:
        print("Create camera constraints based on DLC predictions")
        _, position_funcs = misc.get_pose_state(robot)
        misc.create_camera_contraints(robot,
                                      params,
                                      scene,
                                      dlc_dir,
                                      position_funcs,
                                      hand_labeled_data,
                                      kinetic_dataset=kinetic_dataset)
    else:
        position_funcs = []  # No need for position functions if we are not going to relate back to 2D image.
    total_mass = sum(cast(float, link.mass) for link in robot.links)
    scale_forces_by = total_mass * 9.81
    print(f"Total cheetah mass: {total_mass:.2f} Kg")
    cheetah_estimator = CheetahEstimator(cheetah_name, data_path, robot, params, scene, scale_forces_by, position_funcs,
                                         kinematic_model, enable_eom_slack)
    # Set the solver to be used - IPOPT in this case.
    pe.utils.set_ipopt_path(solver_path)

    return cheetah_estimator


def estimate_kinematics(estimator: CheetahEstimator,
                        solver_output: bool = True,
                        monocular_constraints: bool = False,
                        disable_pose_prior: bool = False,
                        disable_motion_prior: bool = False,
                        pose_model_num_components: int = 5,
                        motion_model_window_size: int = 4,
                        motion_model_sparse_solution: bool = True,
                        out_dir_prefix: Optional[str] = None) -> bool:
    """Estimates the trajectory of the cheetah using the kinematic model of motion.

    Args:
        estimator (CheetahEstimator): Cheetah estimator object that was created using `init_trajectory` method.
        solver_output (bool, optional): Flag that controls the output from the PyoMo solver. Defaults to True.
        monocular_constraints (bool, optional): Flag that enables the extra monocular constraints used to regularise the solution. Defaults to False.
        disable_pose_prior (bool, optional): Flag to disable the pose prior. Defaults to False.
        disable_motion_prior (bool, optional): Flag to disable the motion prior. Defaults to False.
        pose_model_num_components (int, optional): Number of components to use for the pose GMM. Defaults to 5.
        motion_model_window_size (int, optional): The window size used for the linear regression model of motion. Defaults to 4.
        motion_model_sparse_solution (bool, optional): Flag to determine whether a sparse solution (LASSO) or a regular (L2-norm) is used for linear regression. Defaults to True.
        out_dir_prefix (Optional[str], optional): The output directory to save the result. Defaults to None which would save the result in the same location as `root_dir/data_path`.

    Returns:
        bool: Whether the optimisation converged to a satisfactory point.
    """
    # Check that a simple kinematic model of the cheetah is used. Otherwise this process won't produce the best results.
    robot, m = estimator.model, cast(pyo.ConcreteModel, estimator.model.m)
    params = estimator.params
    scene = estimator.scene
    assert not hasattr(robot, "eom_f"), "Cannot use a dynamic model of the cheetah to simply estimate kinematics."
    # Add init of state vector.
    x_est, y_est, z_est, psi_est = misc.create_trajectory_estimate(robot,
                                                                   params,
                                                                   scene,
                                                                   kinetic_dataset=params.kinetic_dataset)
    for fe, cp in robot.indices(one_based=True):
        for link in robot.links:
            for q in link.pyomo_sets["q_set"]:
                robot[link.name]["q"][fe, cp, q].value = 0.0
                robot[link.name]["dq"][fe, cp, q].value = 0.0
                robot[link.name]["ddq"][fe, cp, q].value = 0.0
                robot[link.name]["q"][fe, cp, "psi"].value = float(psi_est[params.start_frame + (fe - 1)])
        robot["base"]["q"][fe, cp, "x"].value = float(x_est[params.start_frame + (fe - 1)])
        robot["base"]["q"][fe, cp, "y"].value = float(y_est[params.start_frame + (fe - 1)])
        robot["base"]["q"][fe, cp, "z"].value = float(z_est[params.start_frame + (fe - 1)])
    # Initialise the 3D pose from the initial estimate.
    misc.init_3d_pose(robot, estimator.position_funcs)
    # Add foot contraints to ensure it does not penetrate the ground plane.
    # misc.create_foot_contraints(robot)
    # Create cost function and run optimisation.
    slack_model_err = misc.constant_acc_cost(robot)
    slack_meas_err = misc.measurement_cost(robot, False, kinetic_dataset=params.kinetic_dataset)
    slack_pose_err = 0
    slack_motion_err = 0
    if monocular_constraints and scene.cam_idx is not None:
        slack_pose_err, pose_model = misc.gmm_pose_cost(robot, pose_model_num_components,
                                                        params.data_dir) if not disable_pose_prior else (0.0, None)
        slack_motion_err, motion_model = misc.add_linear_motion_model(
            robot, motion_model_window_size, motion_model_sparse_solution,
            params.data_dir) if not disable_motion_prior else (0.0, None)
        estimator.pose_model = pose_model
        estimator.motion_model = motion_model
    pe.utils.remove_constraint_if_exists(m, "cost")
    m.cost = pyo.Objective(expr=1e-3 * (slack_meas_err + slack_model_err + slack_pose_err + slack_motion_err))
    estimator.costs = {
        "measurement": slack_meas_err,
        "model": slack_model_err,
        "pose": slack_pose_err,
        "motion": slack_motion_err
    }
    # Solve...
    time0 = time()
    results = pe.utils.default_solver(
        max_mins=180,
        solver="ma97",
        warm_start_init_point=False,
        OF_hessian_approximation="exact",
        Tol=1e-3,
    ).solve(robot.m, tee=solver_output)
    estimator.opt_time_s = time() - time0
    # Print details about the solution state.
    estimator.solution_details()
    # Perform contact detection and save results to a autogen-contacts.json file.
    estimator.com_pos, estimator.com_vel = misc.get_com(robot, scene)
    ret = estimator.is_solution_acceptable(results, 0.1, 0.2)
    if ret:
        # Save optimisation.
        fname = f"fte_kinematic{'_gt' if params.hand_labeled_data else ''}"
        fname = fname if scene.cam_idx is None or monocular_constraints else "fte_kinematic_orig"
        fname = fname if scene.cam_idx is None else f"{fname}_{scene.cam_idx}"
        estimator.save(fname, out_dir_prefix=out_dir_prefix)
    if estimator.scene.cam_idx is not None and monocular_constraints is True:
        # Determine single view error if a single camera used.
        if ret:
            misc.compare_traj_error(params, estimator.scene, include_kinetic=False, out_dir_prefix=out_dir_prefix)

    return ret


def determine_contacts(estimator: CheetahEstimator,
                       monocular: bool = False,
                       verbose: bool = True,
                       out_dir_prefix: Optional[str] = None):
    """Contact detection using the kinematics as a simple heuristic for contact determination.

    Args:
        estimator (CheetahEstimator): Cheetah estimator object that was created using `init_trajectory` method.
        monocular (bool, optional): Flag that determines whether the monocular solution should be used. Defaults to False.
        verbose (bool, optional): Flag to control the output level. Defaults to True.
        out_dir_prefix (Optional[str], optional): The output directory to save the result. Defaults to None which would save the result in the same location as `root_dir/data_path`.
    """
    assert hasattr(estimator.model, "eom_f"), "Dynamic model of the cheetah is required to determine contacts."
    robot, m = estimator.model, cast(pyo.ConcreteModel, estimator.model.m)
    print("Initialise trajectory with previous estimation of kinematics")
    data_dir = estimator.params.data_dir if out_dir_prefix is None else os.path.join(
        out_dir_prefix,
        estimator.params.data_dir.split("cheetah_videos")[1][1:])
    # Init from previous run FTE.
    fte_states = data_ops.load_pickle(
        os.path.join(data_dir, "fte_kinematic" if not monocular else f"fte_kinematic_{estimator.scene.cam_idx}",
                     "fte.pickle"))
    init_q = fte_states["q"]
    init_dq = fte_states["dq"]
    init_ddq = fte_states["ddq"]
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
    # TODO: Should use stance averaged speed instead of stride average speed.
    speed = np.linalg.norm(estimator.com_vel, axis=1)  # type: ignore
    contacts, contacts_tmp = misc.contact_detection(robot,
                                                    estimator.params.start_frame,
                                                    cast(float, np.mean(speed)),
                                                    estimator.scene.fps,
                                                    data_dir,
                                                    plot=verbose)
    avg_vel = np.mean(cast(np.ndarray, estimator.com_vel), axis=0)
    if verbose:
        print("Height, velocity, stance time heuristic:")
        print(contacts)
        print("Height:")
        print(contacts_tmp)
    misc.synth_grf_data(robot, cast(float, np.mean(speed)), 1.0 if avg_vel[0] < 0 else -1.0,
                        os.path.join(data_dir, "grf"))
    misc.synth_grf_data(robot, cast(float, np.mean(speed)), 1.0 if avg_vel[0] < 0 else -1.0,
                        os.path.join(data_dir, "grf"), "autogen-contact-02.json", "data_synth_02")


def estimate_kinetics(estimator: CheetahEstimator,
                      init_torques: bool = True,
                      auto: bool = True,
                      use_2d_reprojections: bool = True,
                      solver_output: bool = True,
                      init_prev_kinematic_solution: bool = True,
                      synthesised_grf: bool = False,
                      no_slip: bool = True,
                      joint_estimation: bool = False,
                      fix_grf: bool = True,
                      ground_constraint: bool = False,
                      disable_pose_prior: bool = False,
                      disable_motion_prior: bool = False,
                      plot: bool = False,
                      out_fname: str = "fte",
                      out_dir_prefix: Optional[str] = None) -> bool:
    """Estimates the trajectory of the cheetah using the dynamic model of motion.

    Args:
        estimator (CheetahEstimator): Cheetah estimator object that was created using `init_trajectory` method.
        init_torques (bool, optional): Flag to estimate the joint torques prior to solving the optimisation problem. Note this requires the `init_prev_kinematic_solution` flag to be True. Defaults to True.
        auto (bool, optional): Flag to determine whether we use contact points that are estimated or not. Defaults to True.
        use_2d_reprojections (bool, optional): Flag to include the 2D reprojections in the cost function as opposed to a 3D kinematic cost from previous kinematic solution. Defaults to True.
        solver_output (bool, optional): Flag that controls the output from the PyoMo solver. Defaults to True.
        init_prev_kinematic_solution (bool, optional): Flag to initialise the kinematic state to the previous solution to `estimate_kinematics` function. Defaults to True.
        synthesised_grf (bool, optional): Flag to indicate if we should use synthesised ground reaction force (GRF). Defaults to False.
        no_slip (bool, optional): Flag to indicate if we should use the no-slip model of contact. Defaults to True.
        joint_estimation (bool, optional): Flag to indicate if we should perform a joint estimation of torques and GRF. Defaults to False.
        fix_grf (bool, optional): Flag to fix the GRF prior to solving the problem. Defaults to True.
        ground_constraint (bool, optional): Flag to ensure the foot is on the ground during a contact phase. Defaults to False.
        disable_pose_prior (bool, optional): Flag to disable the pose prior. Defaults to False.
        disable_motion_prior (bool, optional): Flag to disable the motion prior. Defaults to False.
        plot (bool, optional): Flag to output plots generated during the process. Defaults to False.
        out_fname (str, optional): Filename of the output files. Defaults to "fte".
        out_dir_prefix (Optional[str], optional): The output directory to save the result. Defaults to None which would save the result in the same location as `root_dir/data_path`.

    Returns:
        bool: Whether the optimisation converged to a satisfactory point.
    """
    assert hasattr(estimator.model, "eom_f"), "Dynamic model of the cheetah is required."
    robot, m = estimator.model, cast(pyo.ConcreteModel, estimator.model.m)
    params = estimator.params
    print("Initialise trajectory with previous estimation of kinematics")
    # Init from previous run FTE.
    data_dir = params.data_dir if out_dir_prefix is None else os.path.join(out_dir_prefix,
                                                                           params.data_dir.split("cheetah_videos")[1][1:])
    fte_states = data_ops.load_pickle(
        os.path.join(
            data_dir, "fte_kinematic" if estimator.scene.cam_idx is None or
            (not init_prev_kinematic_solution) else f"fte_kinematic_{estimator.scene.cam_idx}", "fte.pickle"))
    init_q = fte_states["q"]
    init_dq = fte_states["dq"]
    init_ddq = fte_states["ddq"]
    estimator.com_vel = fte_states["com_vel"]
    estimator.com_pos = fte_states["com_pos"]
    avg_vel = np.mean(cast(np.ndarray, estimator.com_vel), axis=0)
    if not auto and (not os.path.exists(os.path.join(estimator.params.data_dir, "grf", "data.h5"))):
        speed = np.linalg.norm(estimator.com_vel, axis=1)  # type: ignore
        misc.synth_grf_data(robot, cast(float, np.mean(speed)), 1.0 if avg_vel[0] < 0 else -1.0,
                            os.path.join(estimator.params.data_dir), "metadata.json", "grf/data")
    if not init_prev_kinematic_solution:
        # Add init of state vector.
        x_est, y_est, z_est, psi_est = misc.create_trajectory_estimate(robot,
                                                                       params,
                                                                       estimator.scene,
                                                                       kinetic_dataset=params.kinetic_dataset)
        for fe, cp in robot.indices(one_based=True):
            for link in robot.links:
                for q in link.pyomo_sets["q_set"]:
                    robot[link.name]["q"][fe, cp, q].value = 0.0
                    robot[link.name]["dq"][fe, cp, q].value = 0.0
                    robot[link.name]["ddq"][fe, cp, q].value = 0.0
                    robot[link.name]["q"][fe, cp, "psi"].value = float(psi_est[params.start_frame + (fe - 1)])
            robot["base"]["q"][fe, cp, "x"].value = float(x_est[params.start_frame + (fe - 1)])
            robot["base"]["q"][fe, cp, "y"].value = float(y_est[params.start_frame + (fe - 1)])
            robot["base"]["q"][fe, cp, "z"].value = float(z_est[params.start_frame + (fe - 1)])
    else:
        for fe, cp in robot.indices(one_based=True):
            p = 0
            for link in robot.links:
                for q in link.pyomo_sets["q_set"]:
                    robot[link.name]["q"][fe, cp, q].value = init_q[fe - 1, p]
                    robot[link.name]["dq"][fe, cp, q].value = init_dq[fe - 1, p]
                    robot[link.name]["ddq"][fe, cp, q].value = init_ddq[fe - 1, p]
                    p += 1

    def add_constraints(name: str, func: Any, indexes: Iterable):
        setattr(m, name, pyo.Constraint(*indexes, rule=func))

    height_uncertainty_m = 0.03 if estimator.params.kinetic_dataset else 0.1
    if joint_estimation:
        with open(os.path.join(data_dir, "grf/autogen-contact.json") if auto else os.path.join(
                params.data_dir, "metadata.json"),
                  "r",
                  encoding="utf-8") as f:
            contact_json = json.load(f)
        start_frame = contact_json["start_frame"]
        foot_contact_order = contact_json["contacts"]
        feet_contacts = []
        for foot in pe.foot.feet(robot):
            foot_vel = foot["gamma"]
            foot_zvel = foot["foot_z_vel"]
            # Add 1 to match pyomo indexing.
            contact_times = []
            if foot_contact_order[foot.name] is not None:
                for f_contact in foot_contact_order[foot.name]:
                    start_contact = (f_contact[0] - start_frame) + 1
                    end_contact = (f_contact[1] - start_frame) + 1
                    contact_times = contact_times + list(range(start_contact, end_contact + 1))
            # Add no slip model that prevents xy velocity of feet when in "contact" with ground.
            add_constraints(f"{foot.name}_no_slipping", lambda m, fe, cp: foot_vel[fe, cp] <= 1
                            if fe in contact_times else pyo.Constraint.Skip, (m.fe, m.cp))
            # TODO: This would be a good thing to add but might make things hard to solve.
            if estimator.params.kinetic_dataset:
                add_constraints(
                    f"{foot.name}_fixed", lambda m, fe, cp: foot_zvel[fe, cp] <= 1
                    if fe in contact_times else pyo.Constraint.Skip, (m.fe, m.cp))
            feet_contacts.append(contact_times)
        print(feet_contacts)
        misc.prescribe_contact_order(robot, feet_contacts, foot_height_uncertainty=height_uncertainty_m)
    else:
        if synthesised_grf:
            grf_z, grf_xy = misc.get_grf_profile(robot,
                                                 params,
                                                 1.0 if avg_vel[0] < 0 else -1.0,
                                                 1 / estimator.scale_forces_by,
                                                 out_dir_prefix=out_dir_prefix,
                                                 synthetic_data=auto)
        else:
            grf_z, grf_xy = estimator.estimate_grf(monocular=True, plot=False, out_dir_prefix=out_dir_prefix)
        estimator.synthesised_grf = grf_z
        for foot in pe.foot.feet(robot):
            grfz = foot["GRFz"]
            grfxy = foot["GRFxy"]
            foot_vel = foot["gamma"]
            foot_zvel = foot["foot_z_vel"]
            foot_height = foot["foot_height"]
            contact_detected = False
            contacts = [0, 0]
            for fe, cp in robot.indices(one_based=True):
                grfz[fe, cp].fix(grf_z[foot.name][fe - 1])
                if ground_constraint and grf_z[foot.name][fe - 1] > 0:
                    # Ensure the foot is more or less on the ground when GRF is acting.
                    foot_height[fe, cp].setub(height_uncertainty_m)
                    foot_height[fe, cp].setlb(-height_uncertainty_m)
                if not fix_grf:
                    # If we unfix the GRF, allow it to move between 20% of the initial estimate.
                    grfz[fe, cp].bounds = misc.bound_value(grf_z[foot.name][fe - 1], 0.2)
                    grfz[fe, cp].fixed = False
                else:
                    # Fixed GRF - so remove the unnecessary friction cone constraint.
                    pe.utils.remove_constraint_if_exists(m, f"{foot.name}_friction_polyhedron_constr")
                for i in range(foot.nsides):
                    grfxy[fe, cp, i].fix(grf_xy[foot.name][fe - 1][i])
                    if not fix_grf:
                        # If we unfix the GRF, allow it to move between 20% of the initial estimate.
                        grfxy[fe, cp, i].bounds = misc.bound_value(grf_xy[foot.name][fe - 1][i], 0.2)
                        grfxy[fe, cp, i].fixed = False
                if not contact_detected and grf_z[foot.name][fe - 1] > 0:
                    contacts[0] = fe - 1
                    contact_detected = True
                if contact_detected and grf_z[foot.name][fe - 1] == 0:
                    contacts[1] = fe
                    contact_detected = False
            if no_slip:
                # Add no slip model that prevents xy velocity of feet when in "contact" with ground.
                add_constraints(
                    f"{foot.name}_no_slipping", lambda m, fe, cp: foot_vel[fe, cp] <= 1
                    if (fe > contacts[0] and fe < contacts[1]) else pyo.Constraint.Skip, (m.fe, m.cp))
                # TODO: This would be a good thing to add but might make things hard to solve.
                if estimator.params.kinetic_dataset:
                    add_constraints(
                        f"{foot.name}_fixed", lambda m, fe, cp: foot_zvel[fe, cp] <= 1
                        if (fe > contacts[0] and fe < contacts[1]) else pyo.Constraint.Skip, (m.fe, m.cp))
    if use_2d_reprojections:
        # Initialise the 3D pose from the initial estimate.
        misc.init_3d_pose(robot, estimator.position_funcs)
    if plot:
        _ = plt.figure(figsize=(16, 9), dpi=120)
        fig_idx = 1
        for foot in pe.foot.feet(robot):
            ax = plt.subplot(2, 2, fig_idx)
            GRFz = pe.utils.get_vals(foot.pyomo_vars["GRFz"], tuple())
            ax.plot(GRFz, label=foot.name)
            ax.set_title(foot.name)
            fig_idx += 1
        plt.show(block=False)
        plt.close()
        _ = plt.figure(figsize=(16, 9), dpi=120)
        fig_idx = 1
        for foot in pe.foot.feet(robot):
            ax = plt.subplot(2, 2, fig_idx)
            GRFxy = pe.utils.get_vals(foot.pyomo_vars["GRFxy"], (foot.pyomo_sets['fric_set'], ))
            GRFxy_mag = np.sum(GRFxy, axis=2)
            ax.plot(GRFxy_mag)
            ax.set_title(foot.name)
            fig_idx += 1
        plt.show(block=False)
        plt.close()
    # Initialise the torque values.
    print("Initialise joint torques.")
    if not init_torques:
        # Set torques to 0.
        for motor in pe.motor.torques(robot):
            for tau in [
                    motor.pyomo_vars["Tc"][fe, idx] for fe in cast(Iterable, m.fe) for idx in motor.pyomo_sets["Tc_set"]
            ]:
                tau.value = 0.0
    else:
        # Initialise torques from the kinemtatic solution and the assumed GRF profile.
        estimator.init_torques()
    # Create cost function and run optimisation.
    # Uncomment the below if you are trying to obtain torque bounds.
    # pe.motor.max_power_constraint(robot, (estimator.scale_forces_by / 9.81) * 0.5 * 600)
    pe.utils.remove_constraint_if_exists(m, "cost")
    if use_2d_reprojections:
        kinematic_cost = misc.measurement_cost(robot, False, kinetic_dataset=params.kinetic_dataset)
        motion_energy = misc.motion_smoothing_cost(robot, estimator.scene.fps)
    else:
        kinematic_cost = misc.kinematic_cost(robot, init_q, estimator.scale_forces_by / 9.81)
        motion_energy = 1e-2 * pe.motor.torque_squared_penalty(robot)
    slack_cost = misc.eom_slack_cost(robot) if estimator.enable_eom_slack else 0.0
    torque_cost = pe.motor.torque_squared_penalty(robot)
    slack_pose_err = 0.0
    if not disable_pose_prior and estimator.scene.cam_idx is not None:
        slack_pose_err, _ = misc.gmm_pose_cost(robot, n_comps=5, data_dir=params.data_dir)
    motion_prior_err = (torque_cost + 0.1 *
                        (estimator.scene.fps)**-2 * motion_energy) if not disable_motion_prior else 0.0
    m.cost = pyo.Objective(expr=1e-3 * (kinematic_cost + slack_pose_err + motion_prior_err + 10e3 * slack_cost))
    estimator.costs = {
        "measurement": kinematic_cost,
        "pose": slack_pose_err,
        "energy": motion_energy,
        "eom_error": slack_cost,
        "torque": torque_cost
    }
    # Uncomment to Print model to file.
    # with open("./pyomo-model.txt", 'w') as output_file:
    #     m.pprint(output_file)
    time0 = time()
    results = pe.utils.default_solver(
        max_mins=300,
        solver="ma97",
        warm_start_init_point=False,
        Tol=1e-3,
        OF_hessian_approximation="exact",
    ).solve(robot.m, tee=solver_output)
    estimator.opt_time_s = time() - time0
    # Print details about the solution state.
    estimator.solution_details()
    # Kinematic fit error.
    base_error, relative_error = estimator.joint_error(init_q[:len(m.fe), :])
    print(f"RMSE base: {base_error:.4f}")
    print(f"RMSE links: {relative_error:.4f}")
    # Update CoM measurements.
    estimator.com_pos, estimator.com_vel = misc.get_com(robot, estimator.scene)
    ret = estimator.is_solution_acceptable(results, base_error, relative_error)
    if estimator.scene.cam_idx is not None or ret:
        # Save optimisation.
        dir_name = f"fte_kinetic{'_gt' if params.hand_labeled_data else ''}"
        dir_name = dir_name if estimator.scene.cam_idx is None else f"{dir_name}_{estimator.scene.cam_idx}"
        estimator.save(dir_name, fname=out_fname, out_dir_prefix=out_dir_prefix)
    if estimator.scene.cam_idx is not None:
        # Determine single view error if a single camera used.
        misc.compare_traj_error(params,
                                estimator.scene,
                                include_kinetic=True,
                                kinetic_out_fname=out_fname,
                                out_dir_prefix=out_dir_prefix)

    return ret


def estimate_grf(estimator: CheetahEstimator, solver_output: bool = True, out_dir_prefix: Optional[str] = None):
    robot, m = estimator.model, cast(pyo.ConcreteModel, estimator.model.m)
    params = estimator.params
    assert estimator.params.kinetic_dataset, "Cannot determine GRF on a dataset other than the kinetic dataset from Penny Hudson and Co."
    # Enable the estimation of contact forces using an acceptable error threshold for the linear complimentarity constraint.
    # for foot in pe.foot.feet(robot):
    #     cast(pe.foot.Foot3D, foot).enable_lcp(True, 0.1)
    print("Initialise trajectory with previous estimation of torques")
    # Init from previous run FTE.
    data_dir = params.data_dir if out_dir_prefix is None else os.path.join(out_dir_prefix,
                                                                           params.data_dir.split("cheetah_videos")[1][1:])
    fte_states = data_ops.load_pickle(os.path.join(data_dir, "fte_kinetic", "fte.pickle"))
    # Initialise the variable timestep from the previous optimisation.
    # prev_cheetah = data_ops.load_pickle(os.path.join(params.data_dir, "fte_kinetic", "cheetah.pickle"))
    init_q = fte_states["q"]
    init_dq = fte_states["dq"]
    init_ddq = fte_states["ddq"]
    init_tau = fte_states["tau"]
    for fe, cp in robot.indices(one_based=True):
        p = 0
        for link in robot.links:
            for q in link.pyomo_sets["q_set"]:
                robot[link.name]["q"][fe, cp, q].value = init_q[fe - 1, p]
                robot[link.name]["dq"][fe, cp, q].value = init_dq[fe - 1, p]
                robot[link.name]["ddq"][fe, cp, q].value = init_ddq[fe - 1, p]
                p += 1
    # Init foot height.
    # misc.init_foot_height(robot)
    # Initialise torques to be fixed from a previous estimation.
    for motor in pe.motor.torques(robot):
        for fe in cast(Iterable, m.fe):
            for idx in motor.pyomo_sets["Tc_set"]:
                # motor.pyomo_vars["Tc"][fe, idx].fix(init_tau[motor.name][fe - 1, idx])
                # Within 10% of the previously estimated torque value.
                motor.pyomo_vars["Tc"][fe, idx].value = init_tau[motor.name][fe - 1, idx]
                motor.pyomo_vars["Tc"][fe, idx].bounds = misc.bound_value(init_tau[motor.name][fe - 1, idx], 0.1)
    # Ensure 0 GRF where no contact is made.
    grf_z, grf_xy = misc.get_grf_profile(robot, params, 1.0, 1 / estimator.scale_forces_by, synthetic_data=False)
    for foot in pe.foot.feet(robot):
        grfz = foot["GRFz"]
        grfxy = foot["GRFxy"]
        foot_height = foot["foot_height"]
        for fe, cp in robot.indices(one_based=True):
            if grf_z[foot.name][fe - 1] == 0:
                # Ensure the foot is more or less on the ground when GRF is acting.
                grfz[fe, cp].fix(0.0)
            if grf_z[foot.name][fe - 1] > 0:
                foot_height[fe, cp].setub(0.03)
                foot_height[fe, cp].setlb(-0.03)
            for i in range(foot.nsides):
                if grf_z[foot.name][fe - 1] == 0:
                    grfxy[fe, cp, i].fix(0.0)
    # Create cost function and run optimisation.
    pe.utils.remove_constraint_if_exists(m, "cost")
    meas_err = misc.measurement_cost(robot, False, kinetic_dataset=params.kinetic_dataset)
    slack_cost = misc.eom_slack_cost(robot)
    torque_cost = pe.motor.torque_squared_penalty(robot)
    motion_energy = misc.motion_smoothing_cost(robot, estimator.scene.fps)
    motion_prior_err = (torque_cost + 0.1 * (estimator.scene.fps)**-2 * motion_energy)
    m.cost = pyo.Objective(expr=1e-3 * (meas_err + motion_prior_err + 10e3 * slack_cost))
    estimator.costs = {"measurement": meas_err, "energy": motion_energy, "eom_error": slack_cost, "torque": torque_cost}
    time0 = time()
    results = pe.utils.default_solver(
        max_mins=300,
        solver="ma97",
        warm_start_init_point=False,
        Tol=1e-3,
        OF_hessian_approximation="exact",
    ).solve(robot.m, tee=solver_output)
    estimator.opt_time_s = time() - time0
    # Print details about the solution state.
    estimator.solution_details()
    # Kinematic fit error.
    base_error, relative_error = estimator.joint_error(init_q)
    print(f"RMSE base: {base_error:.4f}")
    print(f"RMSE links: {relative_error:.4f}")

    ret = estimator.is_solution_acceptable(results, base_error, relative_error)

    if ret:
        estimator.save("fte_grf", fname="fte", out_dir_prefix=out_dir_prefix)

    return ret
