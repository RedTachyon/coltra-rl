import numpy as np

from pycolab import human_ui
from pycolab.things import Backdrop, Drape, Sprite
from pycolab.prefab_parts.sprites import MazeWalker
from pycolab.engine import Engine
from pycolab.plot import Plot

from typing import List, Tuple, Dict, NamedTuple, Union

# CONSTANTS
NORTH = 0
SOUTH = 1
WEST = 2
EAST = 3
STAY = 4

AGENT = 'A'
OBSTACLE = 'O'
EMPTY = ','
GOAL = 'G'

COLOUR_FG = {
    EMPTY:   (400, 400, 400),      # Gray background
    AGENT:  (999, 0,   0),   # Red agent
    OBSTACLE: (999,   999, 999),  # Black obstacle
    GOAL:    (999, 999, 0),   # Yellow goal
}

STEP_REWARD = -0.01
GOAL_REWARD = 1.


BOARD = ",,,,,,,,,,,,,\n" \
        ",,,,,,,,,,,,,\n" \
        ",,,O,,,,,O,,,\n" \
        ",,,O,,,,,O,,,\n" \
        ",,,O,,,,,O,,,\n" \
        ",,,O,,A,,O,,,\n" \
        ",,,O,,,,,O,,,\n" \
        ",,,O,,,,,O,,,\n" \
        ",,,O,,,,,O,,,\n" \
        ",,,OOOOOOO,,,\n" \
        ",,,,,,,,,,,,,\n" \
        ",,,,,,G,,,,,,\n" \
        ",,,,,,,,,,,,,"

class Forager(MazeWalker):
    """
    The agent representation in the Pycolab game.
    Can walk anywhere within the board, collect subgoals and reach the goal
    """

    def __init__(self,
                 corner: NamedTuple,
                 position: NamedTuple,
                 character: str,
                 impassable: str):

        super().__init__(corner, position, character, impassable=impassable, confined_to_board=True)

    def update(self,
               action: int,
               board: np.ndarray,
               layers: Dict[str, np.ndarray],
               backdrop: Backdrop,
               things: Dict[str, Union[Sprite, Drape]],
               the_plot: Plot):

        # Take the action, using methods from the MazeWalker template
        if action == NORTH:
            self._north(board, the_plot)
        elif action == SOUTH:
            self._south(board, the_plot)
        elif action == WEST:
            self._west(board, the_plot)
        elif action == EAST:
            self._east(board, the_plot)
        elif action == STAY:
            self._stay(board, the_plot)
        else:
            the_plot.log(f"Invalid action {action} detected")


class Obstacle(Drape):
    """
    Drape representing the subgoals. They're all stored inside an instance of this object,
    with coordinates stored in the positions argument.
    """

    def __init__(self, curtain: np.ndarray,
                 character: str,
                 positions: List[Tuple[int, int]]):

        super().__init__(curtain, character)

        # Set the initial positions of the subgoals to True
        for (s_row, s_col) in positions:
            self.curtain[s_row, s_col] = True

    def update(self,
               actions: Dict[str, int],
               board: np.ndarray,
               layers: Dict[str, np.ndarray],
               backdrop: Backdrop,
               things: Dict[str, Union[Sprite, Drape]],
               the_plot: Plot):

        pass

class Goal(Sprite):
    """
    Stationary goal, responsible for giving negative rewards with time, and finishing the game.
    """

    def __init__(self, corner, position, character):
        super().__init__(corner, position, character)

    def update(self,
               actions: Dict[str, int],
               board: np.ndarray,
               layers: Dict[str, np.ndarray],
               backdrop: Backdrop,
               things: Dict[str, Union[Sprite, Drape]],
               the_plot: Plot):

        the_plot.add_reward(STEP_REWARD)  # Small negative reward at each time step

        agent_position = things[AGENT].position
        if self.position == agent_position:  # if the first agent collects the goal
            the_plot.log(f"Goal reached")
            the_plot.add_reward(GOAL_REWARD)
            the_plot.terminate_episode()


class Field(Backdrop):
    """
    Backdrop for the game. Doesn't really do anything, but is required by pycolab
    """

    def __init__(self, curtain, palette):
        super().__init__(curtain, palette)

        # Fill the backdrop with a constant value.
        start = np.full_like(curtain, palette[EMPTY], dtype=np.uint8)
        np.copyto(self.curtain, start)


def parse_board(board: str) -> Tuple[Tuple[int, int],
                                     List[Tuple[int, int]],
                                     Tuple[int, int]]:
    """
    Example of board representation:
    A,OO
    O,,O
    OO,,
    OG,,
    It's a single string with newlines
    """
    agent_position = None
    obstacle_positions = []
    goal_position = None

    rows = board.split("\n")
    for i, row in enumerate(rows):
        for j, char in enumerate(row):
            if char == AGENT:
                agent_position = (i, j)
            elif char == OBSTACLE:
                obstacle_positions.append((i, j))
            elif char == GOAL:
                goal_position = (i, j)

    return agent_position, obstacle_positions, goal_position


def create_game(rows: int = 13, cols: int = 13,
                agent_position: Tuple[int, int] = None,
                obstacle_positions: List[Tuple[int, int]] = None,
                goal_position: Tuple[int, int] = None,
                **kwargs) -> Engine:
    """
    Sets up the pycolab foraging game.

    :return: Engine object
    """

    engine = Engine(rows=rows, cols=cols, occlusion_in_layers=False)
    engine.set_backdrop(EMPTY, Field)

    engine.update_group('1. Agents')
    engine.add_sprite(AGENT, agent_position, Forager, impassable=[OBSTACLE])

    engine.update_group('2. Obstacles')
    engine.add_drape(OBSTACLE, Obstacle, positions=obstacle_positions)

    engine.update_group('3. Goal')
    engine.add_sprite(GOAL, goal_position, Goal)

    return engine



if __name__ == '__main__':
    # print(*parse_board(BOARD))
    game = create_game(13, 13, *parse_board(BOARD))

    ui = human_ui.CursesUi(
        keys_to_actions={
                         'w': NORTH,
                         's': SOUTH,
                         'a': WEST,
                         'd': EAST,

                         'f': STAY
                         },
        delay=1000,
        colour_fg=COLOUR_FG,
    )

    ui.play(game)