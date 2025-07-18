from mimicgen.envs.robosuite.coffee import Coffee_D0, CoffeePreparation_D0
from mimicgen.envs.robosuite.stack import Stack_D0
from mimicgen.envs.robosuite.hammer_cleanup import HammerCleanup_D0
from mimicgen.envs.robosuite.three_piece_assembly import ThreePieceAssembly_D0
from mimicgen.envs.robosuite.nut_assembly import NutAssembly_D0
from mimicgen.envs.robosuite.mug_cleanup import MugCleanup_D0, MugCleanup_D0_StageReward
from mimicgen.envs.robosuite.kitchen import Kitchen_D0
from mimicgen.envs.robosuite.nut_assembly import Square_D0
from mimicgen.envs.robosuite.stack import Stack_D0, StackThree_D0
from mimicgen.envs.robosuite.threading import Threading_D0

POS_SETTINGS = {
    "O": (0, 0, 0),  # origin
    "R": (0, 0.2, 0),  # right
    "L": (0, -0.2, 0),  # left
    "LF": (0.1, -0.2, 0),  # left front
    "RF": (0.1, 0.2, 0),  # right front
    "R_R0": (0, 0.2, 0, 0, 0, -0.5),  # right with 1 yaw
    "L_R0": (0, -0.2, 0, 0, 0, 0.5),  # right with 1 yaw
    "R1": (0, 0.1, 0),  
    "L1": (0, -0.1, 0),  
    "R1_R0": (0, 0.1, 0, 0, 0, -0.3),  
    "L1_R0": (0, -0.1, 0, 0, 0, 0.3),  
    "LV": (0.3, -0.5, 0, 0, 0, 1.5),  # left vertical
}

TASKS = [
    Stack_D0,
    HammerCleanup_D0,
    ThreePieceAssembly_D0,
    NutAssembly_D0,
    MugCleanup_D0,
    Kitchen_D0,
    Coffee_D0,
    CoffeePreparation_D0,
    Square_D0,
    StackThree_D0,
    Threading_D0,
    
    MugCleanup_D0_StageReward,
]


def create_env(class_derived, class_name, func=None):
    assert class_derived is not None and class_name is not None
    cls_func = {}
    if func is not None:
        cls_func = func
    return type(class_name, (class_derived,), cls_func)


def gen_func(offsets: tuple):
    def _set_robot(self):
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](
            self.table_full_size[0]
        )
        x, y, z = xpos
        xpos = (x + offsets[0], y + offsets[1], z + offsets[2])
        self.robots[0].robot_model.set_base_xpos(xpos)
        if len(offsets) > 3:
            rotation = (offsets[3], offsets[4], offsets[5])
            self.robots[0].robot_model.set_base_ori(rotation)

    return {"_set_robot": _set_robot}


for task in TASKS:
    for pos in POS_SETTINGS:
        create_env(task, task.__name__ + "_" + pos, gen_func(POS_SETTINGS[pos]))
