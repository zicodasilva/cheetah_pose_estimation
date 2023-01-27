from typing import Any, Dict, Iterable, List, Optional, Tuple, Callable
from typing_extensions import Literal
from math import pi as π
from sympy import Matrix as Mat
import numpy as np
from pyomo.environ import Objective
import pyomo.environ as pyo

from shared.physical_education.links import Link3D, constrain_rel_angle
from shared.physical_education.system import System3D
from shared.physical_education.foot import add_foot, feet, Foot3D, feet_penalty
from shared.physical_education.motor import add_torque, torque_squared_penalty
from shared.physical_education.drag import add_drag
from shared.physical_education.spring import add_torquespring
from shared.physical_education.damper import add_torquedamper
from shared.physical_education.utils import remove_constraint_if_exists


def define_leg(params: Dict[str, Any],
               body: Link3D,
               front: bool,
               right: bool,
               signed_axis_aligned: Literal[1, -1] = 1,
               include_dynamics: bool = True,
               relative_orientation: bool = False) -> Iterable[Link3D]:
    """Define a leg and attach it to the front/back right/left of `body`.
            Only really makes sense when `body` is aligned along the `x`-axis"""

    # maybe flip x (or y)
    # the model is considered to face along the x axis (so front/back
    # refers to changes in the y value).
    def mfx(x):
        return -signed_axis_aligned * x if front else signed_axis_aligned * x

    def mfy(y):
        return y if right else -y

    start_I = body.Pb_I + body.Rb_I @ Mat([mfx(body.length / 2), mfy(body.radius), 0])
    suffix = ("F" if front else "B") + ("R" if right else "L")

    frontorback_str = "front" if front else "back"
    rightorleft_str = "right" if right else "left"
    p = params[frontorback_str]
    if relative_orientation:
        thigh = Link3D("U" + suffix,
                       "-z",
                       start_I=start_I,
                       **p["thigh"],
                       meta=["leg", "thigh", frontorback_str, rightorleft_str],
                       rotations="y",
                       parent_orientation=body.Rb_I)
        calf = Link3D("L" + suffix,
                      "-z",
                      start_I=thigh.bottom_I,
                      **p["calf"],
                      meta=["leg", "calf", frontorback_str, rightorleft_str],
                      rotations="y",
                      parent_orientation=thigh.Rb_I)
    else:
        thigh = Link3D("U" + suffix,
                       "-z",
                       start_I=start_I,
                       **p["thigh"],
                       meta=["leg", "thigh", frontorback_str, rightorleft_str])
        calf = Link3D("L" + suffix,
                      "-z",
                      start_I=thigh.bottom_I,
                      **p["calf"],
                      meta=["leg", "calf", frontorback_str, rightorleft_str])
        # input torques: hip pitch and abduct
        body.add_revolute_joint(thigh, about="y")
        thigh.add_revolute_joint(calf, about="y")
    # next, all of the muscles and their respective limits
    muscleparams = params["motor"][frontorback_str]
    if include_dynamics:
        # add_torque(body,
        #         thigh,
        #         name=f"{frontorback_str}-{rightorleft_str}-hip-abduct",
        #         about="x",
        #         **muscleparams["hip-abduct"])
        add_torque(body,
                   thigh,
                   name=f"{frontorback_str}-{rightorleft_str}-hip-pitch",
                   about="y",
                   **muscleparams["hip-pitch"])
        add_torque(thigh, calf, about="y", **muscleparams["knee"])
    if relative_orientation:
        hock = Link3D("H" + suffix,
                      "-z",
                      start_I=calf.bottom_I,
                      **p["hock"],
                      meta=["leg", "calf", frontorback_str, rightorleft_str],
                      rotations="y",
                      parent_orientation=calf.Rb_I)
    else:
        hock = Link3D("H" + suffix,
                      "-z",
                      start_I=calf.bottom_I,
                      **p["hock"],
                      meta=["leg", "calf", frontorback_str, rightorleft_str])
        calf.add_revolute_joint(hock, about="y")
    if include_dynamics:
        add_torque(calf, hock, about="y", **muscleparams["ankle"])
        add_foot(hock, at="bottom", nsides=4, GRFxy_max=3.0, GRFz_max=3.0, friction_coeff=params["friction_coeff"])

    return thigh, calf, hock


def model(params: Dict[str, Any],
          include_dynamics: bool = True,
          relative_orientation: bool = False) -> Tuple[System3D, Callable[[System3D], None]]:
    # create neck/head base link along with the front and back body links.
    body_aligned = "-x"
    body_B = Link3D("base", body_aligned, base=True, **params["body_B"], meta=["spine", "back"])
    if relative_orientation:
        body_F = Link3D("bodyF",
                        body_aligned,
                        start_I=body_B.bottom_I,
                        **params["body_F"],
                        meta=["spine", "front"],
                        rotations="yz",
                        parent_orientation=body_B.Rb_I)
        neck = Link3D("neck",
                      body_aligned,
                      start_I=body_F.bottom_I,
                      rotations="yz",
                      parent_orientation=body_F.Rb_I,
                      **params["neck"])
        if include_dynamics:
            add_torque(neck, body_F, about="yz", **params["motor"]["neck"])
            add_torque(body_F, body_B, about="yz", **params["motor"]["spine"])

        # Setup tail.
        tail0 = Link3D("tail0",
                       body_aligned,
                       start_I=body_B.top_I,
                       **params["tail0"],
                       meta=["tail"],
                       rotations="yz",
                       parent_orientation=body_B.Rb_I)
        tail1 = Link3D("tail1",
                       body_aligned,
                       start_I=tail0.bottom_I,
                       **params["tail1"],
                       meta=["tail"],
                       rotations="yz",
                       parent_orientation=tail0.Rb_I)
    else:
        body_F = Link3D("bodyF", body_aligned, start_I=body_B.bottom_I, **params["body_F"], meta=["spine", "front"])
        neck = Link3D("neck", body_aligned, start_I=body_F.bottom_I, **params["neck"])
        # neck.add_hookes_joint(body_F, about="yz")
        # body_F.add_hookes_joint(body_B, about="yz")
        if include_dynamics:
            add_torque(neck, body_F, about="xyz", **params["motor"]["neck"])
            add_torque(body_F, body_B, about="xyz", **params["motor"]["spine"])

        # Setup tail.
        tail0 = Link3D("tail0", "+x", start_I=body_B.top_I, **params["tail0"], meta=["tail"])
        tail1 = Link3D("tail1", "+x", start_I=tail0.bottom_I, **params["tail1"], meta=["tail"])
        body_B.add_hookes_joint(tail0, about="yz")
        tail0.add_hookes_joint(tail1, about="yz")

    if include_dynamics:
        add_torque(body_B, tail0, about="yz", **params["motor"]["spine-tail0"])
        add_torque(tail0, tail1, about="yz", **params["motor"]["tail0-tail1"])
    # Setup each leg.
    ufl, lfl, hfl = define_leg(params,
                               body_F,
                               front=True,
                               right=False,
                               signed_axis_aligned=1 if body_aligned == "-x" else -1,
                               include_dynamics=include_dynamics,
                               relative_orientation=relative_orientation)
    ufr, lfr, hfr = define_leg(params,
                               body_F,
                               front=True,
                               right=True,
                               signed_axis_aligned=1 if body_aligned == "-x" else -1,
                               include_dynamics=include_dynamics,
                               relative_orientation=relative_orientation)
    ubl, lbl, hbl = define_leg(params,
                               body_B,
                               front=False,
                               right=False,
                               signed_axis_aligned=1 if body_aligned == "-x" else -1,
                               include_dynamics=include_dynamics,
                               relative_orientation=relative_orientation)
    ubr, lbr, hbr = define_leg(params,
                               body_B,
                               front=False,
                               right=True,
                               signed_axis_aligned=1 if body_aligned == "-x" else -1,
                               include_dynamics=include_dynamics,
                               relative_orientation=relative_orientation)

    # combine into a robot
    robot = System3D("3D quadruped",
                     [body_B, body_F, neck, tail0, tail1, ufl, lfl, hfl, ufr, lfr, hfr, ubl, lbl, ubr, lbr, hbl, hbr])

    return robot, add_pyomo_constraints


def add_pyomo_constraints(robot: System3D, relative_angles: bool = False, kinetic_dataset: bool = False) -> None:
    # π/3 = 60 degrees
    # π/2 = 90 degrees
    # π/4 = 45 degrees
    assert robot.m is not None,\
        "robot does not have a pyomo model defined on it"

    body_B, body_F, neck, tail0, tail1, \
    ufl, lfl, hfl, ufr, lfr, hfr, \
    ubl, lbl, ubr, lbr, \
    hbl, hbr = [link["q"] for link in robot.links]

    if relative_angles:
        ncp = len(robot.m.cp)
        #===== POSE CONSTRAINTS (Note 1 based indexing for pyomo!!!!...@#^!@#&) =====
        # Head
        robot.m.head_theta_0 = pyo.Constraint(robot.m.fe,
                                              rule=lambda m, n: (-np.pi / 6, neck[n, ncp, "theta"], np.pi / 6))
        if kinetic_dataset:
            robot.m.head_phi_0 = pyo.Constraint(robot.m.fe, rule=lambda m, n: (-0.1, neck[n, ncp, "phi"], 0.1))
            robot.m.head_psi_0 = pyo.Constraint(robot.m.fe,
                                                rule=lambda m, n: neck[n - 1, ncp, "psi"] == neck[n, ncp, "psi"]
                                                if n > 1 else pyo.Constraint.Skip)
        else:
            robot.m.head_phi_0 = pyo.Constraint(robot.m.fe,
                                                rule=lambda m, n: (-np.pi / 6, neck[n, ncp, "phi"], np.pi / 6))
        # Neck
        # robot.m.neck_phi_1 = pyo.Constraint(m.N, rule=lambda m, n: (-np.pi / 2, m.x[n, _pyo_i(idx["phi_1"])], np.pi / 2))
        # robot.m.neck_theta_1 = pyo.Constraint(m.N, rule=lambda m, n: (-np.pi / 6, m.x[n, _pyo_i(idx["theta_1"])], np.pi / 6))
        # robot.m.neck_psi_1 = pyo.Constraint(m.N, rule=lambda m, n: (-np.pi / 6, m.x[n, _pyo_i(idx["psi_1"])], np.pi / 6))
        # Front torso
        robot.m.front_torso_theta_2 = pyo.Constraint(robot.m.fe,
                                                     rule=lambda m, n: (-np.pi / 6, body_F[n, ncp, "theta"], np.pi / 6))
        if kinetic_dataset:
            robot.m.front_torso_psi_2 = pyo.Constraint(robot.m.fe, rule=lambda m, n: (-0.1, body_F[n, ncp, "psi"], 0.1))
        else:
            robot.m.front_torso_psi_2 = pyo.Constraint(robot.m.fe,
                                                       rule=lambda m, n: (-np.pi / 6, body_F[n, ncp, "psi"], np.pi / 6))
        # Back torso
        robot.m.back_torso_theta_3 = pyo.Constraint(robot.m.fe,
                                                    rule=lambda m, n: (-np.pi / 6, body_B[n, ncp, "theta"], np.pi / 6))
        if kinetic_dataset:
            # robot.m.back_torso_phi_3 = pyo.Constraint(robot.m.fe, rule=lambda m, n: (-np.pi / 6, body_B[n, ncp, "phi"], np.pi / 6))
            robot.m.back_torso_psi_3 = pyo.Constraint(robot.m.fe, rule=lambda m, n: (-0.1, body_B[n, ncp, "psi"], 0.1))
        else:
            # robot.m.back_torso_phi_3 = pyo.Constraint(robot.m.fe, rule=lambda m, n: (-0.1, body_B[n, ncp, "phi"], 0.1))
            robot.m.back_torso_psi_3 = pyo.Constraint(robot.m.fe,
                                                      rule=lambda m, n: (-np.pi / 6, body_B[n, ncp, "psi"], np.pi / 6))
        # Tail base
        robot.m.tail_base_theta_4 = pyo.Constraint(robot.m.fe,
                                                   rule=lambda m, n: (-(2 / 3) * np.pi, tail0[n, ncp, "theta"],
                                                                      (2 / 3) * np.pi))
        robot.m.tail_base_psi_4 = pyo.Constraint(robot.m.fe,
                                                 rule=lambda m, n: (-(2 / 3) * np.pi, tail0[n, ncp, "psi"],
                                                                    (2 / 3) * np.pi))
        # # Tail mid
        robot.m.tail_mid_theta_5 = pyo.Constraint(robot.m.fe,
                                                  rule=lambda m, n: (-(2 / 3) * np.pi, tail1[n, ncp, "theta"],
                                                                     (2 / 3) * np.pi))
        robot.m.tail_mid_psi_5 = pyo.Constraint(robot.m.fe,
                                                rule=lambda m, n: (-(2 / 3) * np.pi, tail1[n, ncp, "psi"],
                                                                   (2 / 3) * np.pi))
        # Front left leg
        robot.m.l_shoulder_theta_6 = pyo.Constraint(robot.m.fe,
                                                    rule=lambda m, n: (-(3 / 4) * np.pi, ufl[n, ncp, "theta"],
                                                                       (3 / 4) * np.pi))
        robot.m.l_front_knee_theta_7 = pyo.Constraint(robot.m.fe, rule=lambda m, n: (-np.pi, lfl[n, ncp, "theta"], 0))
        # Front right leg
        robot.m.r_shoulder_theta_8 = pyo.Constraint(robot.m.fe,
                                                    rule=lambda m, n: (-(3 / 4) * np.pi, ufr[n, ncp, "theta"],
                                                                       (3 / 4) * np.pi))
        robot.m.r_front_knee_theta_9 = pyo.Constraint(robot.m.fe, rule=lambda m, n: (-np.pi, lfr[n, ncp, "theta"], 0))
        # Back left leg
        robot.m.l_hip_theta_10 = pyo.Constraint(robot.m.fe,
                                                rule=lambda m, n: (-(3 / 4) * np.pi, ubl[n, ncp, "theta"],
                                                                   (3 / 4) * np.pi))
        robot.m.l_back_knee_theta_11 = pyo.Constraint(robot.m.fe, rule=lambda m, n: (0, lbl[n, ncp, "theta"], np.pi))
        # Back right leg
        robot.m.r_hip_theta_12 = pyo.Constraint(robot.m.fe,
                                                rule=lambda m, n: (-(3 / 4) * np.pi, ubr[n, ncp, "theta"],
                                                                   (3 / 4) * np.pi))

        robot.m.r_back_knee_theta_13 = pyo.Constraint(robot.m.fe, rule=lambda m, n: (0, lbr[n, ncp, "theta"], np.pi))
        # robot.m.l_front_ankle_theta_14 = pyo.Constraint(robot.m.fe,
        #                                                 rule=lambda m, n: (-(3 / 4) * np.pi, hfl[n, ncp, "theta"],
        #                                                                    np.pi / 4))
        # robot.m.r_front_ankle_theta_15 = pyo.Constraint(robot.m.fe,
        #                                                 rule=lambda m, n: (-np.pi / 4, hfr[n, ncp, "theta"],
        #                                                                    (3 / 4) * np.pi))
        # robot.m.l_back_ankle_theta_16 = pyo.Constraint(robot.m.fe,
        #                                                rule=lambda m, n: (0, hbl[n, ncp, "theta"], (3 / 4) * np.pi))
        # robot.m.r_back_ankle_theta_17 = pyo.Constraint(robot.m.fe,
        #                                                rule=lambda m, n: (0, hbr[n, ncp, "theta"], (3 / 4) * np.pi))
        robot.m.l_front_ankle_theta_14 = pyo.Constraint(robot.m.fe,
                                                        rule=lambda m, n: (-np.pi / 4, hfl[n, ncp, "theta"],
                                                                           (3 / 4) * np.pi))
        robot.m.r_front_ankle_theta_15 = pyo.Constraint(robot.m.fe,
                                                        rule=lambda m, n: (-np.pi / 4, hfr[n, ncp, "theta"],
                                                                           (3 / 4) * np.pi))
        robot.m.l_back_ankle_theta_16 = pyo.Constraint(robot.m.fe,
                                                       rule=lambda m, n: (-(3 / 4) * np.pi, hbl[n, ncp, "theta"], 0))
        robot.m.r_back_ankle_theta_17 = pyo.Constraint(robot.m.fe,
                                                       rule=lambda m, n: (-(3 / 4) * np.pi, hbr[n, ncp, "theta"], 0))
    else:
        ncp = len(robot.m.cp)
        if kinetic_dataset:
            constrain_rel_angle(robot.m, "neck_yaw", -0.05, neck[:, :, "psi"], body_F[:, :, "psi"], 0.05)
            constrain_rel_angle(robot.m, "neck_roll", -0.05, neck[:, :, "phi"], body_F[:, :, "phi"], 0.05)
            robot.m.head_yaw_straight_run = pyo.Constraint(
                robot.m.fe,
                rule=lambda m, n: neck[n - 1, ncp, "psi"] == neck[n, ncp, "psi"] if n > 1 else pyo.Constraint.Skip)
            robot.m.spine_phi_0 = pyo.Constraint(robot.m.fe, rule=lambda m, n: (-0.05, body_B[n, ncp, "phi"], 0.05))
            constrain_rel_angle(robot.m, "spine_yaw", -0.1, body_F[:, :, "psi"], body_B[:, :, "psi"], 0.1)
            constrain_rel_angle(robot.m, "spine_roll", -0.1, body_F[:, :, "phi"], body_B[:, :, "phi"], 0.1)
            constrain_rel_angle(robot.m, "tail_body_yaw", -0.1, body_B[:, :, "psi"], tail0[:, :, "psi"], 0.1)
        else:
            constrain_rel_angle(robot.m, "neck_yaw", -π / 6, neck[:, :, "psi"], body_F[:, :, "psi"], π / 6)
            constrain_rel_angle(robot.m, "neck_roll", -π / 6, neck[:, :, "phi"], body_F[:, :, "phi"], π / 6)
            robot.m.spine_phi_0 = pyo.Constraint(robot.m.fe, rule=lambda m, n: (-π / 6, body_B[n, ncp, "phi"], π / 6))
            constrain_rel_angle(robot.m, "spine_yaw", -π / 6, body_F[:, :, "psi"], body_B[:, :, "psi"], π / 6)
            constrain_rel_angle(robot.m, "spine_roll", -π / 6, body_F[:, :, "phi"], body_B[:, :, "phi"], π / 6)
            constrain_rel_angle(robot.m, "tail_body_yaw", -π / 1.5, body_B[:, :, "psi"], tail0[:, :, "psi"], π / 1.5)
        constrain_rel_angle(robot.m, "neck_pitch", -π / 6, neck[:, :, "theta"], body_F[:, :, "theta"], π / 6)
        # spine can"t bend too much:
        constrain_rel_angle(robot.m, "spine_pitch", -π / 6, body_F[:, :, "theta"], body_B[:, :, "theta"], π / 6)
        # tail can"t go too crazy:
        constrain_rel_angle(robot.m, "tail_body_pitch", -π / 1.5, body_B[:, :, "theta"], tail0[:, :, "theta"], π / 1.5)
        constrain_rel_angle(robot.m, "tail_tail_pitch", -π / 1.5, tail0[:, :, "theta"], tail1[:, :, "theta"], π / 1.5)
        constrain_rel_angle(robot.m, "tail_tail_yaw", -π / 1.5, tail0[:, :, "psi"], tail1[:, :, "psi"], π / 1.5)

        # legs: hip abduction and knee
        for body, thigh, calf, hock, name in ((body_F, ufl, lfl, hfl, "FL"), (body_F, ufr, lfr, hfr, "FR"),
                                              (body_B, ubl, lbl, hbl, "BL"), (body_B, ubr, lbr, hbr, "BR")):
            constrain_rel_angle(robot.m, name + "_hip_pitch", -0.75 * π, body[:, :, "theta"], thigh[:, :, "theta"],
                                0.75 * π)
            # constrain_rel_angle(robot.m, name + "_hip_aduct", -π / 8, body[:, :, "phi"], thigh[:, :, "phi"], π / 8)
            # lo, up = (-π / 1.5, 0) if name.startswith("B") else (0, π / 1.5)
            # if kinetic_dataset:
            lo, up = (0, π) if name.startswith("B") else (-π, 0)
            # else:
            #     lo, up = (-π / 1.5, 0) if name.startswith("B") else (-π / 1.5, 0)
            constrain_rel_angle(robot.m, name + "_knee", lo, thigh[:, :, "theta"], calf[:, :, "theta"], up)
            # lo, up = (-0)
            # lo, up = (0, π / 1.5) if name.startswith("B") else (-π / 6, π / 6)
            # if kinetic_dataset:
            lo, up = (-0.75 * π, 0) if name.startswith("B") else (-π / 4, 0.75 * π)
            # else:
            #     lo, up = (0, +π / 1.75) if name.startswith("B") else (-π / 3, π / 3)
            # lo, up = (0, +π / 1.5) if name.startswith("B") else (-π / 1.5, +π / 4)
            constrain_rel_angle(robot.m, name + "_foot", lo, calf[:, :, "theta"], hock[:, :, "theta"], up)
            # lo, up = (0, +π / 1.5) if name.startswith("B") else (-π / 1.5, +π / 4)
            # for th in hock[:, :, "theta"]:
            #     th.setub(up)
            #     th.setlb(lo)


# common functions
def high_speed_stop(robot: System3D,
                    initial_vel: float,
                    minimize_distance: bool,
                    gallop_data: Optional[dict] = None,
                    offset: int = 0):
    import math
    import random
    from shared.physical_education.utils import copy_state_init
    from shared.physical_education.init_tools import add_costs

    nfe = len(robot.m.fe)
    ncp = len(robot.m.cp)
    total_time = float((nfe - 1) * robot.m.hm0.value)

    body = robot["base"]

    # start at the origin
    body["q"][1, ncp, "x"].fix(0)
    body["q"][1, ncp, "y"].fix(0)

    if gallop_data is not None:
        for fed, cpd in robot.indices(one_based=True):
            robot.init_from_dict_one_point(gallop_data,
                                           fed=fed,
                                           cpd=cpd,
                                           fes=(fed - 1 + offset) % nfe,
                                           cps=0,
                                           skip_if_fixed=True,
                                           skip_if_not_None=False,
                                           fix=False)

        for link in robot.links:
            for q in link.pyomo_sets["q_set"]:
                link["q"][1, ncp, q].fixed = True
                link["dq"][1, ncp, q].fixed = True
    else:
        # init to y plane
        body["q"][:, :, "y"].value = 0
        for link in robot.links:
            for ang in ("phi", "psi"):
                link["q"][:, :, ang].value = 0
                link["dq"][:, :, ang].value = 0
                link["ddq"][:, :, ang].value = 0

        # roughly bound to y plane
        for fe, cp in robot.indices(one_based=True):
            body["q"][fe, cp, "y"].setub(0.2)
            body["q"][fe, cp, "y"].setlb(-0.2)

        for link in robot.links:
            for ang in ("phi", "psi"):
                for fe, cp in robot.indices(one_based=True):
                    link["q"][fe, cp, ang].setub(math.pi / 4)
                    link["q"][fe, cp, ang].setlb(-math.pi / 4)

        # bound theta
        for fe, cp in robot.indices(one_based=True):
            for link in robot.links[4:]:  # all leg segments - no tail or body
                link["q"][fe, cp, "theta"].setub(math.radians(60))
                link["q"][fe, cp, "theta"].setlb(math.radians(-60))
            for link in robot.links[1:3]:  # two body segments
                link["q"][fe, cp, "theta"].setub(math.radians(45))
                link["q"][fe, cp, "theta"].setlb(math.radians(-45))

        for link in robot.links:
            for fe, cp in robot.indices(one_based=True):
                link["q"][fe, cp, "theta"].value = (math.radians(random.gauss(0, 15)))

        body["q"][1, ncp, "z"].fix(0.6)

        # both sides mirrored
        for src, dst in (("UFL", "UFR"), ("LFL", "LFR"), ("UBL", "UBR"), ("LBL", "LBR"), ("HBL", "HBR"), ("HFL",
                                                                                                          "HFR")):
            copy_state_init(robot[src]["q"], robot[dst]["q"])

        # init tail to flick?
        for link in robot.links[3:5]:
            for fe, cp in robot.indices(one_based=True):
                link["q"][fe, cp, "theta"].value = (math.radians(random.random() * 60))

        # stop weird local minimum where it bounces
        for fe, cp in robot.indices(one_based=True):
            if fe in range(10):
                continue
            # if fe > nfe/2: continue

            height = body["q"][fe, cp, "z"]
            height.setub(0.6)  # approx. leg height

            for foot in feet(robot):
                foot["foot_height"][fe, cp].setub(0.01)

    # start at speed
    body["dq"][1, ncp, "x"].fix(initial_vel)

    # end at rest
    for link in robot.links:
        for q in link.pyomo_sets["q_set"]:
            link["dq"][nfe, ncp, q].fix(0)

    # end in a fairly standard position
    for link in robot.links[1:3]:  # two body segments
        link["q"][nfe, ncp, "theta"].setub(math.radians(10))
        link["q"][nfe, ncp, "theta"].setlb(math.radians(-10))
    for link in robot.links[5:]:  # leaving out tail - it might flail, which is good
        link["q"][nfe, ncp, "theta"].setub(math.radians(20))
        link["q"][nfe, ncp, "theta"].setlb(math.radians(-20))

    for link in robot.links:
        for ang in ("phi", "psi"):
            link["q"][nfe, ncp, ang].setub(math.radians(5))
            link["q"][nfe, ncp, ang].setlb(math.radians(-5))

    # position and velocity over time
    for fe in robot.m.fe:
        pos = total_time * (initial_vel / 2) * (fe - 1) / (nfe - 1)
        vel = initial_vel * (1 - (fe - 1) / (nfe - 1))
        # print("pos", pos, "vel", vel)
        body["q"][fe, :, "x"].value = pos
        body["dq"][fe, :, "x"].value = vel

    # objective
    distance_cost = body["q"][nfe, ncp, "x"] if minimize_distance else 0
    return add_costs(robot,
                     include_transport_cost=False,
                     include_torque_cost=False,
                     distance_cost=0.0001 * distance_cost)


def periodic_gallop_test(robot: System3D,
                         avg_vel: float,
                         feet: Iterable["Foot3D"],
                         foot_order_vals: Iterable[Tuple[int, int]],
                         init_from_dict: Optional[dict] = None,
                         at_angle_d: Optional[float] = None):
    """
    foot_order_vals = ((1, 7), (6, 13), (31, 38), (25, 32))  # 14 m/s
    """
    from math import sin, cos, radians
    import random
    from shared.physical_education import utils
    from shared.physical_education.foot import prescribe_contact_order
    from shared.physical_education.init_tools import sin_around_touchdown, add_costs
    from shared.physical_education.constrain import straight_leg, periodic

    nfe = len(robot.m.fe)
    ncp = len(robot.m.cp)
    m = utils.get_pyomo_model_or_error(robot)
    total_time = utils.total_time(m)
    utils.constrain_total_time(m, total_time=total_time)

    body = robot["body"]

    # start at the origin
    body["q"][1, ncp, "x"].fix(0)
    body["q"][1, ncp, "y"].fix(0)

    if init_from_dict is None:
        if at_angle_d is None or at_angle_d == 0:
            # init to y plane
            body["q"][:, :, "y"].value = 0

        # running in a straight line
        for link in robot.links:
            for ang in ("phi", "psi"):
                link["q"][:, :, ang].value = (radians(at_angle_d or 0) if ang == "psi" else 0)
                link["dq"][:, :, ang].value = 0
                link["ddq"][:, :, ang].value = 0

        for fe, cp in robot.indices(one_based=True):
            var = robot.links[0]["q"][fe, cp, "psi"]
            var.setub(radians((at_angle_d or 0) + 10))
            var.setlb(radians((at_angle_d or 0) - 10))

        # init theta
        def rand(mu, sigma, offset=0):
            return radians(random.gauss(mu, sigma) + offset)

        for fe, cp in robot.indices(one_based=True):
            # body
            robot.links[0]["q"][fe, cp, "theta"].value = rand(0, 15)
            robot.links[1]["q"][fe, cp, "theta"].value = rand(0, 15, +10)
            # tail
            robot.links[2]["q"][fe, cp, "theta"].value = rand(0, 15, -10)
            robot.links[3]["q"][fe, cp, "theta"].value = rand(0, 15, -10)

        for link in robot.links[5:]:  # legs
            for fe, cp in robot.indices(one_based=True):
                link["q"][fe, cp, "theta"].value = rand(0, 30)

        # body height
        body["q"][:, :, "z"].value = 0.55

        # the feet:
        prescribe_contact_order(feet, foot_order_vals)
        for (touchdown, liftoff), foot in zip(foot_order_vals, [foot.name.rstrip("_foot") for foot in feet]):
            lower, upper = foot, "U" + foot[1:]
            straight_leg(robot[upper]["q"], robot[lower]["q"], [touchdown], state="theta")

            angles = sin_around_touchdown(int((touchdown + liftoff) / 2), len(robot.m.fe))
            for fe, val in zip(robot.m.fe, angles):  # type: ignore
                robot[upper]["q"][fe, :, "theta"].value = val
                robot[lower]["q"][fe, :, "theta"].value = val + \
                    radians(-15 if upper[1] == "F" else 15)

        # get timestep bounds ready
        # [long/short] timesteps in the air
        robot.m.hm[:].value = robot.m.hm[1].lb
        for start, stop in foot_order_vals:
            for fe in range(start, stop + 1):
                # but [short/long] timesteps while on the ground
                robot.m.hm[fe].value = robot.m.hm[fe].ub
    else:
        if init_from_dict["ncp"] == 1:
            for fed, cpd in robot.indices(one_based=True):
                robot.init_from_dict_one_point(init_from_dict,
                                               fed=fed,
                                               cpd=cpd,
                                               fes=fed - 1,
                                               cps=0,
                                               skip_if_fixed=True,
                                               skip_if_not_None=False,
                                               fix=False)
        else:
            robot.init_from_dict(init_from_dict)
        if not (at_angle_d == 0 or at_angle_d is None):
            raise ValueError(f"TODO: rotate init! Got at_angle_d = {at_angle_d}")

    for link in robot.links:
        for fe, cp in robot.indices(one_based=True):
            phi = link["q"][fe, cp, "phi"]
            phi.setub(radians(+15))
            phi.setlb(radians(-15))

            psi = link["q"][fe, cp, "psi"]
            psi.setub(radians(+10 + (at_angle_d or 0)))
            psi.setlb(radians(-10 + (at_angle_d or 0)))

    # bound theta
    # stop the back from going so high!
    for link in robot.links[:3]:  # body
        for fe, cp in robot.indices(one_based=True):
            link["q"][fe, cp, "theta"].setub(radians(+45))
            link["q"][fe, cp, "theta"].setlb(radians(-45))

    for link in robot.links[3:]:  # everything else
        for fe, cp in robot.indices(one_based=True):
            link["q"][fe, cp, "theta"].setub(radians(+90))
            link["q"][fe, cp, "theta"].setlb(radians(-90))

    # never fallen over
    for fe, cp in robot.indices(one_based=True):
        body["q"][fe, cp, "z"].setlb(0.3)
        body["q"][fe, cp, "z"].setub(0.7)

    if at_angle_d is None:
        # roughly bound to y plane
        for fe, cp in robot.indices(one_based=True, skipfirst=False):
            body["q"][fe, cp, "y"].setub(0.2)
            body["q"][fe, cp, "y"].setlb(-0.2)

        # average velocity init (overwrite the init!)
        for fe, cp in robot.indices(one_based=True, skipfirst=False):
            body["q"][fe, cp, "x"].value = avg_vel * \
                total_time * (fe-1 + (cp-1)/ncp)/(nfe-1)
            body["dq"][fe, cp, "x"].value = avg_vel

        body["q"][nfe, ncp, "x"].fix(total_time * avg_vel)

        # periodic
        periodic(robot, but_not=("x", ))
    else:
        θᵣ = radians(at_angle_d)

        # average velocity init (overwrite the init!)
        for fe, cp in robot.indices(one_based=True, skipfirst=False):
            scale = total_time * (fe - 1 + (cp - 1) / ncp) / (nfe - 1)
            body["q"][fe, cp, "x"].value = avg_vel * scale * cos(θᵣ)
            body["dq"][fe, cp, "x"].value = avg_vel * cos(θᵣ)
            body["q"][fe, cp, "y"].value = avg_vel * scale * sin(θᵣ)
            body["dq"][fe, cp, "y"].value = avg_vel * sin(θᵣ)

        #ol.visual.warn("Should probably also bound x, y!")

        body["q"][nfe, ncp, "x"].fix(total_time * avg_vel * cos(θᵣ))
        body["q"][nfe, ncp, "y"].fix(total_time * avg_vel * sin(θᵣ))

        # periodic
        periodic(robot, but_not=("x", "y"))

    return add_costs(robot, include_transport_cost=False, include_torque_cost=False)


def drop_test(robot, *, z_rot: float, min_torque: bool, initial_height: float = 1.) -> Dict[str, Any]:
    """Params which have been tested for this task:
    nfe = 20, total_time = 1.0, vary_timestep_with=(0.8,1.2), 5 mins for solving

    if min_torque is True, quite a bit more time is needed as IPOPT refines things
    """
    nfe = len(robot.m.fe)
    ncp = len(robot.m.cp)
    body = robot["base"]

    # start at the origin
    body["q"][1, ncp, "x"].fix(0)
    body["q"][1, ncp, "y"].fix(0)
    body["q"][1, ncp, "z"].fix(initial_height)

    # fix initial angle
    for link in robot.links:
        for ang in ("phi", "theta"):
            link["q"][1, ncp, ang].fix(0)

        link["q"][1, ncp, "psi"].fix(z_rot)

    # start stationary
    for link in robot.links:
        for q in link.pyomo_sets["q_set"]:
            link["dq"][1, ncp, q].fix(0)

    # init to y plane
    for link in robot.links:
        for ang in ("phi", "theta"):
            link["q"][:, :, ang].value = 0

        link["q"][:, :, "psi"].value = z_rot

    # legs slightly forward at the end
    uplopairs = (("UFL", "LFL"), ("UFR", "LFR"), ("UBL", "LBL"), ("UBR", "LBR"))
    for upper, lower in uplopairs:
        ang = 0.01 if not upper[1] == "B" else -0.01
        robot[upper]["q"][nfe, ncp, "theta"].setlb(ang)
        robot[lower]["q"][nfe, ncp, "theta"].setub(-ang)

    # but not properly fallen over
    body["q"][nfe, ncp, "z"].setlb(0.2)

    # objective: reduce CoT, etc
    remove_constraint_if_exists(robot.m, "cost")

    torque_cost = torque_squared_penalty(robot)
    pen_cost = feet_penalty(robot)
    robot.m.cost = Objective(expr=(torque_cost if min_torque else 0) + 1000 * pen_cost)

    return {"torque": torque_cost, "penalty": pen_cost}
