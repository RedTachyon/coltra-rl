from __future__ import annotations

from itertools import cycle
from dataclasses import dataclass, astuple
import json
import os
import re
from typing import List, Tuple, Dict, Optional

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


###################################
### Data reading and processing ###
###################################

COLORS = [
    (0, 0, 0),  # black
    (0, 0, 1),  # blue
    (0, 1, 1),  # cyan
    (0, 1, 0),  # green
    (1, 0, 1),  # magenta
    (1, 0, 0),  # red
    (1, 0.92, 0.016),  # yellow
    (1, 0.5, 80 / 255),  # coral
    (139 / 255, 0, 139 / 255),  # dark magenta
    (40 / 255, 79 / 255, 79 / 255),  # dark slate gray
    (1, 215 / 255, 0),  # gold
    (1, 182 / 255, 193 / 255),  # light pink
]


def split_ana(content: str) -> List[str]:
    """
    Splits an .ana file into parts, each of which is a string that corresponds to some data batch.
    """
    outputs = []
    temp = []
    for line in content.split("\n"):
        if len(line) > 0 and line[0] != " ":
            outputs.append("\n".join(temp))
            temp = [line]
        else:
            temp.append(line)

    outputs.append("\n".join(temp))
    return outputs[1:]


def parse_segment(content: str) -> Tuple[str, np.ndarray]:
    """Parses a segment of .ana data.
    The first line is assumed to be the title, each line after that has one or more numbers"""
    result = []
    lines = content.split("\n")
    name = lines[0]
    for line in lines[1:-1]:
        line = line.strip()
        numbers = [float(num) for num in line.split(" ") if len(num) > 0]

        result.append(numbers)

    result = np.array(result)
    return name, result


def parse_ana(content: str) -> Dict:
    """Parse the text of the entire file, split it into segments and return a dictionary of arrays"""
    segments = split_ana(content)
    data = [parse_segment(segment) for segment in segments]
    data_dict = {name.strip(): array for name, array in data}
    return data_dict


def read_ana(path: str) -> Dict:
    """Same as read_ana, but handle reading the file as well"""
    with open(path, "r") as f:
        text = f.read()

    data = parse_ana(text)
    return data


def convert_to_json(path: str, out_path: str):
    """Loads the ana file, saves it elsewhere as json. At the moment it only saves the times and positions."""
    data = read_ana(path)
    num_peds, idx = re.findall(r"(\d+)peds-(\d+).ana", path)[0]

    num_peds, idx = int(num_peds), int(idx)

    arrays = []
    for i in range(1, num_peds + 1):
        arrays.append(data[f"center_{i}"])
    positions = np.array(arrays).tolist()

    time = data["time(s)"].ravel().tolist()

    info = {"time": time, "position": positions}

    with open(out_path, "w") as f:
        json.dump(info, f)


def convert_all(base_path: str, out_base_path: str):
    """Convert all ana files to json"""
    files = [file for file in os.listdir(base_path) if file.endswith(".ana")]
    for file in files:
        path = os.path.join(base_path, file)
        out_path = os.path.join(out_base_path, file[:-4] + ".json")
        print(f"Converting {path} to {out_path}")
        convert_to_json(path, out_path)


def read_json(path: str) -> Dict[str, np.ndarray]:
    """Loads the json and converts values into numpy"""
    with open(path, "r") as f:
        data = json.load(f)
    return {key: np.array(value, dtype=np.float32) for key, value in data.items()}


def read_trajectory(path: str, max_time: int = 1000) -> Trajectory:
    traj_dict = read_json(path)
    return Trajectory(
        time=traj_dict["time"][:max_time],
        pos=traj_dict["position"][:, :max_time, :],
        goal=traj_dict["goal"] if "goal" in traj_dict else None,
        finish=traj_dict["finish"].astype(int) if "finish" in traj_dict else None,
    )


####################################
######## Data visualization ########
####################################


def moving_average(a: np.ndarray, n: int = 3) -> np.ndarray:
    ret = np.cumsum(a, axis=-1)
    ret[:, n:] = ret[:, n:] - ret[:, :-n]
    return ret[:, n - 1 :] / n


@dataclass
class Trajectory:
    time: np.ndarray
    pos: np.ndarray
    goal: np.ndarray
    finish: np.ndarray | None

    def __post_init__(self):
        if self.goal is None:
            self.goal = self.pos[:, -1, :]
        if self.finish is None:
            self.finish = np.full(self.pos.shape[0], self.pos.shape[1], dtype=int)

    def __getitem__(self, item):
        if isinstance(item, tuple):
            if len(item) == 1:
                goal_idx = item
                time_idx = slice(None)
                finish_idx = item
            elif len(item) == 2:
                goal_idx, _ = item
                time_idx = item[1]
                finish_idx = item[0]
            else:
                goal_idx = (item[0], item[2])
                time_idx = item[1]
                finish_idx = item[0]
        else:  # isinstance(item, int):
            goal_idx = item
            time_idx = slice(None)
            finish_idx = item
        return Trajectory(
            time=self.time[time_idx],
            pos=self.pos[item],
            goal=self.goal[goal_idx],
            finish=self.finish[finish_idx],
        )

    def __iter__(self):
        return iter(astuple(self))


def get_velocity(trajectory: Trajectory) -> np.ndarray:
    time, pos, goal = trajectory.time, trajectory.pos, trajectory.goal

    velocity = np.diff(pos, axis=-2) / np.diff(time)[None, :, None]

    return velocity


def get_angle(trajectory: Trajectory) -> np.ndarray:
    # TODO: Fix this, it doesn't work due to the sampling frequency in simulated data
    # TODO: also implement Menger curvature

    vel = get_velocity(trajectory)
    norm_vel = vel / np.linalg.norm(vel, axis=-1, keepdims=True)

    v_1 = norm_vel[:, 1:, :]
    v_0 = norm_vel[:, :-1, :]

    costheta = np.clip(np.einsum("ati,ati->at", v_0, v_1), -1, 1)
    theta = np.arccos(costheta)

    return theta


def get_speed(trajectory: Trajectory) -> np.ndarray:
    time, pos, goal = trajectory.time, trajectory.pos, trajectory.goal

    speed = np.diff(pos, axis=-2)
    speed = np.linalg.norm(speed, axis=-1) / np.diff(time)

    return speed


def draw_trajectory(trajectory: Trajectory):
    time, positions, goal = trajectory.time, trajectory.pos, trajectory.goal
    if len(positions.shape) == 2:
        positions = np.expand_dims(positions, 0)

    for agent_traj, c in zip(positions, cycle(COLORS)):
        plt.plot(*agent_traj.T, c=c)

    for pair, c in zip(positions[:, 0, :], cycle(COLORS)):
        plt.scatter(*pair, color=c)


def draw_speed(trajectory: Trajectory, **kwargs):
    speed = get_speed(trajectory).ravel()
    sns.histplot(speed, **kwargs)


def plot_time_profile(time: np.ndarray, x: np.ndarray, mode: str = "minmax"):
    assert mode in ["minmax", "std", "ste"]

    mean = x.mean(0)
    time = time[: x.shape[1]]
    plt.plot(time, mean)

    if mode == "minmax":
        low = x.min(0)
        high = x.max(0)
    elif mode == "std":
        std = x.std(0)
        low = mean - std
        high = mean + std
    elif mode == "ste":
        ste = x.std(0) / np.sqrt(x.shape[0])
        low = mean - ste
        high = mean + ste
    else:
        raise ValueError("This shouldn't happen")

    plt.fill_between(time, low, high, alpha=0.2)


def plot_velocity_profile(trajectory: Trajectory, mode: str = "minmax"):
    speed = get_speed(trajectory)

    plot_time_profile(trajectory.time, speed, mode)


def plot_acceleration_profile(trajectory: Trajectory, mode: str = "minmax"):
    accel = get_acceleration(trajectory, norm=True)

    plot_time_profile(trajectory.time, accel, mode)


def rot_matrix(theta: np.ndarray) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def norm_trajectory(trajectory: Trajectory, fix_sign: bool = False) -> Trajectory:
    time, pos, goal = trajectory.time, trajectory.pos, trajectory.goal

    start = pos[:, 0, np.newaxis, :]
    end = pos[:, -1, np.newaxis, :]

    center_pos = pos - start
    center_end = end - start

    theta = np.arctan2(center_end[..., 1], center_end[..., 0])

    norm_pos = np.einsum("ijb,bti->btj", rot_matrix(theta.ravel()), center_pos)

    if fix_sign:
        sign = np.sign(norm_pos.mean(axis=1, keepdims=True))
        norm_pos *= sign

    norm_traj = Trajectory(
        time=time, pos=norm_pos, goal=norm_pos[:, -1, :], finish=trajectory.finish
    )

    return norm_traj


def get_acceleration(trajectory: Trajectory, norm: bool = False) -> np.ndarray:
    time, pos, goal = trajectory.time, trajectory.pos, trajectory.goal

    vel = get_velocity(trajectory)
    accel = np.diff(vel, axis=-2) / np.diff(time)[None, :-1, None]
    if norm:
        accel = np.linalg.norm(accel, axis=-1)
    return accel


def draw_histogram(
    x: np.ndarray, n_bins: int = 50, xmin: float = 0.0, xmax: float = 30.0, **kwargs
):
    vals, bins = np.histogram(x, bins=n_bins, range=(xmin, xmax))

    plt.bar(bins[:-1], vals, width=(xmax - xmin) / n_bins, align="edge", **kwargs)


def plot_acceleration_histogram(trajectory: Trajectory, **kwargs):
    accel = get_acceleration(trajectory)

    draw_histogram(accel, **kwargs)


def set_size(*args):
    plt.rcParams["figure.figsize"] = args


def make_dashboard(
    data: Trajectory, size: int = 3, show: bool = False, save_path: Optional[str] = None
):
    # UNIT_SIZE = size
    GRID = (4, 8)

    # set_size(8 * UNIT_SIZE, 4 * UNIT_SIZE)
    velocity = get_speed(data)

    accel = get_acceleration(data, norm=True)
    accel = moving_average(accel, 10)

    ################
    ### POSITION ###
    ################

    # Trajectory
    ax = plt.subplot2grid(GRID, (0, 0), rowspan=2, colspan=2)

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    draw_trajectory(data)

    plt.title("Trajectory")

    # Normalized trajectory
    ax = plt.subplot2grid(GRID, (2, 0), rowspan=1, colspan=2)

    draw_trajectory(norm_trajectory(data, fix_sign=False))
    plt.ylim(-3, 3)

    plt.title("Normalized trajectory")

    # Flipped normalized trajectory
    ax = plt.subplot2grid(GRID, (3, 0), rowspan=1, colspan=2)

    draw_trajectory(norm_trajectory(data, fix_sign=True))
    plt.ylim(-3, 3)

    plt.title("Normalized trajectory (same side)")

    ################
    ### VELOCITY ###
    ################

    ax = plt.subplot2grid(GRID, (0, 2), rowspan=2, colspan=3)

    # plot_velocity_profile(data, use_ste=True)
    plot_time_profile(data.time, velocity)
    plt.ylim(0, 2.1)
    plt.xlim(0, 8)
    plt.title("Velocity over time")

    ax = plt.subplot2grid(GRID, (0, 5), rowspan=2, colspan=3)

    # draw_speed(data, bins=20)
    draw_histogram(velocity, n_bins=20, xmin=1e-5, xmax=2.1)
    plt.xlim(0, 2.1)

    plt.title("Velocity histogram")

    ####################
    ### ACCELERATION ###
    ####################

    ax = plt.subplot2grid(GRID, (2, 2), rowspan=2, colspan=3)

    plot_time_profile(data.time, accel)
    plt.xlim(0, 8)
    plt.ylim(0, 10)

    plt.title("Acceleration over time")

    ax = plt.subplot2grid(GRID, (2, 5), rowspan=2, colspan=3)

    xmax = 20
    draw_histogram(accel, xmin=1e-5, xmax=xmax, n_bins=20)
    # plt.yscale('log')
    # plt.ylim(0.5, 2e3)
    plt.xlim(0, xmax)
    plt.title("Acceleration histogram")

    ##############
    ### RENDER ###
    ##############

    plt.tight_layout()
    if show:
        plt.show()

    if save_path is not None:
        plt.savefig(save_path)


# def load_trajnet_data(path: str) -> Trajectory:
#     # Generated by copilot - probably won't work
#     with open(path, "r") as f:
#         data = json.load(f)
#
#     time = np.array(data["timestamps"])
#     pos = np.array(data["positions"])
#     pos = pos[..., :2]
#
#     return Trajectory(time=time, pos=pos)
