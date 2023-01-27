import dill
import shared.physical_education as pe
import acinoset_opt as opt
import cheetah_params
import cheetah

cheetah_names = ["jules", "phantom"]
# cheetah_names = ["phantom"]
for name in cheetah_names:
    robot_name = f"cheetah-{name}-spine-base-kinematics"
    include_dynamics = False
    relative_orientation = False
    print(f"Creating model for {robot_name}")
    # Make your model
    robot, add_pyomo_constraints = cheetah.model(cheetah_params.parameters[name],
                                                 include_dynamics=include_dynamics,
                                                 relative_orientation=relative_orientation)
    if include_dynamics:
        robot.calc_eom(simp_func=lambda x: pe.utils.parsimp(x, nprocs=16))
    # kinetic_dataset = False
    # trial = "run"
    # date = "2019_03_07"
    # root_dir = "/data/zico/cheetah_videos"
    # data_path = f"kinetic_dataset/{date}/{name}/trial{trial}" if kinetic_dataset else f"{date}/{name}/{trial}"
    # estimator = opt.init_trajectory(root_dir,
    #                                 data_path,
    #                                 name,
    #                                 kinetic_dataset,
    #                                 kinematic_model=False,
    #                                 monocular_enable=False,
    #                                 enable_eom_slack=False,
    #                                 shutter_delay_estimation=False,
    #                                 enable_ppm=False)
    # estimator.calc_grf_eom()
    # # Then, eventually, save the symbolic model.
    with open(f"./models/{robot_name}_tmp.robot", "wb") as f:
        # Save the `add_pyomo_constraints` function, because it must correspond to the
        # model and you may change the source in between saving and loading. A potential source of bugs
        dill.dump([robot, add_pyomo_constraints], f)
