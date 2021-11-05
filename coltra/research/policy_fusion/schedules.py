from types import SimpleNamespace

SCHEDULE_0 = [  # Empty schedule
    # Format:
    #  (time, do_clone, [freeze], name, coefficient)
]

Schedules = SimpleNamespace(
    CrowdSchedules=[
        [
            (1, False, [False], "collision", 0.0),
            (100, True, [True, False], "collision", -0.6),
            (500, False, [False, False], "collision", -0.3),
        ],
        [
            (1, False, [False], "collision", 0.0),
            (100, True, [True, False], "collision", -0.3),
            (500, False, [False, False], "collision", -0.3),
        ],
        [
            (1, False, [False], "collision", 0.0),
            (100, True, [True, False], "collision", -0.9),
        ],
    ],
    SmartNavSchedules=[
        [
            (1, False, [False], "visible_reward", 0.0),
            (300, True, [True, False], "visible_reward", -0.005),
            (600, False, [False, False], "visible_reward", -0.001),
        ]
    ],
)
