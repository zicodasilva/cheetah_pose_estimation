from math import pi

parameters = {
    "arabia": {
        "source": """
        Female cheetah with 30Kg body mass.
        https://journals.biologists.com/jeb/article/215/14/2425/10852/High-speed-galloping-in-the-cheetah-Acinonyx - body mass/lengths
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3077521/ - forelimb
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3077520/ - hindlimb
        Combined with figures from "mean" female cheetahs that share more or less the same body mass
        https://academic.oup.com/jmammal/article/84/3/840/905900:
        From "Quasi-steady state aerodynamics of the cheetah tail"
            fur length on tail = 10mm on average
            average tail diameter (no fur) = 31mm
                ---> radius = 31/2 + 10 = 25.5mm = 0.0255m
        Friction coeff of 1.3 from
        "Locomotion dynamics of hunting in wild cheetahs"
        """,
        "neck": {
            "mass": 0.4,
            "radius": 0.1,  # 0.13 for the diameter of the skull and muzzle girth is 0.26.
            "length": 0.218 + 0.09  # skull length + neck length
        },
        "body_F": {
            "mass": 9.0,
            "radius": 0.673 / (2 * pi),
            "length": 0.378
        },
        "body_B": {
            "mass": 18.0,
            "radius": 0.54 / (2 * pi),
            "length": 0.252
        },
        "tail0": {
            "mass": 0.4,
            "radius": 0.0255,
            "length": 0.30
        },
        "tail1": {
            "mass": 0.2,
            "radius": 0.0255,
            "length": 0.30
        },
        "front": {
            "thigh": {
                "mass": 0.162,
                "radius": 0.012,
                "length": 0.242
            },
            "calf": {
                "mass": 0.067,
                "radius": 0.008,
                "length": 0.232
            },
            "hock": {
                "mass": 0.02,
                "radius": 0.008,
                "length": 0.1
            },
        },
        "back": {
            "thigh": {
                "mass": 0.189,
                "radius": 0.012,
                "length": 0.267
            },
            "calf": {
                "mass": 0.156,
                "radius": 0.01,
                "length": 0.278
            },
            "hock": {
                "mass": 0.06,
                "radius": 0.01,
                "length": 0.17
            },
        },
        "friction_coeff": 1.3,
        # measured in terms of body weight
        "motor": {
            "neck": {
                "torque_bounds": (-2, 2),
            },
            "spine": {
                "torque_bounds": (-2, 2),
            },
            "spine-tail0": {
                "torque_bounds": (-2, 2),
            },
            "tail0-tail1": {
                "torque_bounds": (-2, 2),
            },
            "front": {
                "hip-pitch": {
                    "torque_bounds": (-2, 2),
                },
                "hip-abduct": {
                    "torque_bounds": (-2, 2),
                },
                "knee": {
                    "torque_bounds": (-2, 2),
                },
                "ankle": {
                    "torque_bounds": (-2, 2),
                },
            },
            "back": {
                "hip-pitch": {
                    "torque_bounds": (-2, 2),
                },
                "hip-abduct": {
                    "torque_bounds": (-2, 2),
                },
                "knee": {
                    "torque_bounds": (-2, 2),
                },
                "ankle": {
                    "torque_bounds": (-2, 2),
                },
            },
        },
    },
    "shiraz": {
        "source": """
        Cheetah with 35Kg body mass.
        https://journals.biologists.com/jeb/article/215/14/2425/10852/High-speed-galloping-in-the-cheetah-Acinonyx - body mass/lengths
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3077521/ - forelimb
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3077520/ - hindlimb
        Combined with figures from "mean" female cheetahs that share more or less the same body mass
        https://academic.oup.com/jmammal/article/84/3/840/905900:
        From "Quasi-steady state aerodynamics of the cheetah tail"
            fur length on tail = 10mm on average
            average tail diameter (no fur) = 31mm
                ---> radius = 31/2 + 10 = 25.5mm = 0.0255m
        Friction coeff of 1.3 from
        "Locomotion dynamics of hunting in wild cheetahs"
        """,
        "neck": {
            "mass": 0.4,
            "radius": 0.1,  # 0.13 for the diameter of the skull and muzzle girth is 0.26.
            "length": 0.218 + 0.09  # skull length + neck length
        },
        "body_F": {
            "mass": 13.0,
            "radius": 0.673 / (2 * pi),
            "length": 0.378
        },
        "body_B": {
            "mass": 19.0,
            "radius": 0.54 / (2 * pi),
            "length": 0.252
        },
        "tail0": {
            "mass": 0.4,
            "radius": 0.0255,
            "length": 0.30
        },
        "tail1": {
            "mass": 0.2,
            "radius": 0.0255,
            "length": 0.30
        },
        "front": {
            "thigh": {
                "mass": 0.162,
                "radius": 0.012,
                "length": 0.242
            },
            "calf": {
                "mass": 0.067,
                "radius": 0.008,
                "length": 0.232
            },
            "hock": {
                "mass": 0.02,
                "radius": 0.008,
                "length": 0.12
            },
        },
        "back": {
            "thigh": {
                "mass": 0.189,
                "radius": 0.012,
                "length": 0.267
            },
            "calf": {
                "mass": 0.156,
                "radius": 0.01,
                "length": 0.278
            },
            "hock": {
                "mass": 0.06,
                "radius": 0.01,
                "length": 0.17
            },
        },
        "friction_coeff": 1.3,
        # measured in terms of body weight
        "motor": {
            "neck": {
                "torque_bounds": (-2, 2),
            },
            "spine": {
                "torque_bounds": (-2, 2),
            },
            "spine-tail0": {
                "torque_bounds": (-2, 2),
            },
            "tail0-tail1": {
                "torque_bounds": (-2, 2),
            },
            "front": {
                "hip-pitch": {
                    "torque_bounds": (-2, 2),
                },
                "hip-abduct": {
                    "torque_bounds": (-2, 2),
                },
                "knee": {
                    "torque_bounds": (-2, 2),
                },
                "ankle": {
                    "torque_bounds": (-2, 2),
                },
            },
            "back": {
                "hip-pitch": {
                    "torque_bounds": (-2, 2),
                },
                "hip-abduct": {
                    "torque_bounds": (-2, 2),
                },
                "knee": {
                    "torque_bounds": (-2, 2),
                },
                "ankle": {
                    "torque_bounds": (-2, 2),
                },
            },
        },
    },
    "phantom": {
        "source": """
        Cheetah with 32-34Kg body mass.
        """,
        "neck": {
            "mass": 0.4,
            "radius": 0.1,  # 0.13 for the diameter of the skull and muzzle girth is 0.26.
            "length": 0.31
        },
        "body_F": {
            "mass": 12.4,
            "radius": 0.717 / (2 * pi),
            "length": 0.444
        },
        "body_B": {
            "mass": 18.6,
            "radius": 0.594 / (2 * pi),
            "length": 0.296
        },
        "tail0": {
            "mass": 0.4,
            "radius": 0.0255,
            "length": 0.28
        },
        "tail1": {
            "mass": 0.2,
            "radius": 0.0255,
            "length": 0.36
        },
        "front": {
            "thigh": {
                "mass": 0.2052,
                "radius": 0.012,
                "length": 0.26
            },
            "calf": {
                "mass": 0.0816,
                "radius": 0.005,
                "length": 0.27
            },
            "hock": {
                "mass": 0.02,
                "radius": 0.008,
                "length": 0.125
            },
        },
        "back": {
            "thigh": {
                "mass": 0.252,
                "radius": 0.012,
                "length": 0.26
            },
            "calf": {
                "mass": 0.12,
                "radius": 0.01,
                "length": 0.29
            },
            "hock": {
                "mass": 0.072,
                "radius": 0.01,
                "length": 0.265
            },
        },
        "friction_coeff": 1.3,
        # measured in terms of body weight
        "motor": {
            "neck": {
                "torque_bounds": (-2, 2),
            },
            "spine": {
                "torque_bounds": (-2, 2),
            },
            "spine-tail0": {
                "torque_bounds": (-2, 2),
            },
            "tail0-tail1": {
                "torque_bounds": (-2, 2),
            },
            "front": {
                "hip-pitch": {
                    "torque_bounds": (-2, 2),
                },
                "hip-abduct": {
                    "torque_bounds": (-2, 2),
                },
                "knee": {
                    "torque_bounds": (-2, 2),
                },
                "ankle": {
                    "torque_bounds": (-2, 2),
                },
            },
            "back": {
                "hip-pitch": {
                    "torque_bounds": (-2, 2),
                },
                "hip-abduct": {
                    "torque_bounds": (-2, 2),
                },
                "knee": {
                    "torque_bounds": (-2, 2),
                },
                "ankle": {
                    "torque_bounds": (-2, 2),
                },
            },
        },
    },
    "jules": {
        "source": """
        Cheetah with 36-38Kg body mass.
        """,
        "neck": {
            "mass": 0.4,
            "radius": 0.1,  # 0.13 for the diameter of the skull and muzzle girth is 0.26.
            "length": 0.35
        },
        "body_F": {
            "mass": 14.0,
            "radius": 0.717 / (2 * pi),
            "length": 0.444
        },
        "body_B": {
            "mass": 21.0,
            "radius": 0.594 / (2 * pi),
            "length": 0.296
        },
        "tail0": {
            "mass": 0.4,
            "radius": 0.0255,
            "length": 0.28
        },
        "tail1": {
            "mass": 0.2,
            "radius": 0.0255,
            "length": 0.36
        },
        "front": {
            "thigh": {
                "mass": 0.2052,
                "radius": 0.012,
                "length": 0.24
            },
            "calf": {
                "mass": 0.0816,
                "radius": 0.005,
                "length": 0.28
            },
            "hock": {
                "mass": 0.02,
                "radius": 0.008,
                "length": 0.155
            },
        },
        "back": {
            "thigh": {
                "mass": 0.252,
                "radius": 0.012,
                "length": 0.27
            },
            "calf": {
                "mass": 0.12,
                "radius": 0.01,
                "length": 0.33
            },
            "hock": {
                "mass": 0.072,
                "radius": 0.01,
                "length": 0.245
            },
        },
        "friction_coeff": 1.3,
        # measured in terms of body weight
        "motor": {
            "neck": {
                "torque_bounds": (-2, 2),
            },
            "spine": {
                "torque_bounds": (-2, 2),
            },
            "spine-tail0": {
                "torque_bounds": (-2, 2),
            },
            "tail0-tail1": {
                "torque_bounds": (-2, 2),
            },
            "front": {
                "hip-pitch": {
                    "torque_bounds": (-2, 2),
                },
                "hip-abduct": {
                    "torque_bounds": (-2, 2),
                },
                "knee": {
                    "torque_bounds": (-2, 2),
                },
                "ankle": {
                    "torque_bounds": (-2, 2),
                },
            },
            "back": {
                "hip-pitch": {
                    "torque_bounds": (-2, 2),
                },
                "hip-abduct": {
                    "torque_bounds": (-2, 2),
                },
                "knee": {
                    "torque_bounds": (-2, 2),
                },
                "ankle": {
                    "torque_bounds": (-2, 2),
                },
            },
        },
    },
    "acinoset": {
        "source": """
        Cheetah with 44Kg body mass, a general cheetah used for AcinoSet.
        """,
        "neck": {
            "mass": 0.4,
            "radius": 0.1,  # 0.13 for the diameter of the skull and muzzle girth is 0.26.
            "length": 0.218 + 0.09  # skull length + neck length
        },
        "body_F": {
            "mass": 14.0,
            "radius": 0.717 / (2 * pi),
            "length": 0.37
        },
        "body_B": {
            "mass": 28.0,
            "radius": 0.594 / (2 * pi),
            "length": 0.37
        },
        "tail0": {
            "mass": 0.4,
            "radius": 0.0255,
            "length": 0.28
        },
        "tail1": {
            "mass": 0.2,
            "radius": 0.0255,
            "length": 0.36
        },
        "front": {
            "thigh": {
                "mass": 0.171 * 1.2,
                "radius": 0.012,
                "length": 0.24
            },
            "calf": {
                "mass": 0.068 * 1.2,
                "radius": 0.005,
                "length": 0.28
            },
            "hock": {
                "mass": 0.02,
                "radius": 0.008,
                "length": 0.14
            },
        },
        "back": {
            "thigh": {
                "mass": 0.210 * 1.2,
                "radius": 0.012,
                "length": 0.32
            },
            "calf": {
                "mass": 0.100 * 1.2,
                "radius": 0.01,
                "length": 0.25
            },
            "hock": {
                "mass": 0.060 * 1.2,
                "radius": 0.01,
                "length": 0.22
            },
        },
        "friction_coeff": 1.3,
        # measured in terms of body weight
        "motor": {
            "neck": {
                "torque_bounds": (-2, 2),
            },
            "spine": {
                "torque_bounds": (-2, 2),
            },
            "spine-tail0": {
                "torque_bounds": (-2, 2),
            },
            "tail0-tail1": {
                "torque_bounds": (-2, 2),
            },
            "front": {
                "hip-pitch": {
                    "torque_bounds": (-2, 2),
                },
                "hip-abduct": {
                    "torque_bounds": (-2, 2),
                },
                "knee": {
                    "torque_bounds": (-2, 2),
                },
                "ankle": {
                    "torque_bounds": (-2, 2),
                },
            },
            "back": {
                "hip-pitch": {
                    "torque_bounds": (-2, 2),
                },
                "hip-abduct": {
                    "torque_bounds": (-2, 2),
                },
                "knee": {
                    "torque_bounds": (-2, 2),
                },
                "ankle": {
                    "torque_bounds": (-2, 2),
                },
            },
        },
    },
}
