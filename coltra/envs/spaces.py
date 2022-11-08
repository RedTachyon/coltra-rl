from typing import Optional

from gymnasium.spaces import Space, Box, Discrete

from coltra.buffers import Observation, Action


class ObservationSpace(Space):
    vector: Optional[Box] = None
    image: Optional[Box] = None
    rays: Optional[Box] = None
    buffer: Optional[Box] = None

    def __init__(self, spaces: Optional[dict[str, Space]] = None, /, **kwargs: Space):
        super().__init__()
        if spaces is None:
            spaces = kwargs

        self.spaces = spaces

        for name, space in self.spaces.items():
            setattr(self, name, space)

    def sample(self):
        return Observation(
            **{name: space.sample() for name, space in self.spaces.items()}
        )

    def contains(self, x: Observation):
        for name, space in self.spaces.items():
            if not space.contains(getattr(x, name, None)):
                return False
        return True

    def __getitem__(self, item):
        return self.spaces[item]

    def __repr__(self) -> str:
        return (
            "Obs("
            + ", ".join([str(k) + ": " + str(s) for k, s in self.spaces.items()])
            + ")"
        )


class ActionSpace(Space):
    continuous: Optional[Box] = None
    discrete: Optional[Discrete] = None

    def __init__(self, fields: dict[str, Space]):
        super().__init__()
        self.spaces = fields

        for name, space in self.spaces.items():
            setattr(self, name, space)

        # TODO: Ugly fix because I only use a single type of actions
        self.space = list(self.spaces.values())[0]

    def sample(self):
        return Action(**{name: space.sample() for name, space in self.spaces.items()})

    def contains(self, x: Observation):
        for name, space in self.spaces.items():
            if not space.contains(getattr(x, name, None)):
                return False
        return True

    def __repr__(self) -> str:
        return (
            "Action("
            + ", ".join([str(k) + ": " + str(s) for k, s in self.spaces.items()])
            + ")"
        )
