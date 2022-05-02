from typing import Optional

from gym.spaces import Space, Box, Discrete

from coltra.buffers import Observation, Action


class ObservationSpace(Space):
    vector: Optional[Box] = None
    image: Optional[Box] = None
    rays: Optional[Box] = None
    buffer: Optional[Box] = None

    def __init__(self, fields: dict[str, Space]):
        super().__init__()
        self.fields = fields

        for name, space in self.fields.items():
            setattr(self, name, space)

    def sample(self):
        return Observation(
            **{name: space.sample() for name, space in self.fields.items()}
        )

    def contains(self, x: Observation):
        for name, space in self.fields.items():
            if not space.contains(getattr(x, name, None)):
                return False
        return True

    def __repr__(self) -> str:
        return (
            "Obs("
            + ", ".join([str(k) + ": " + str(s) for k, s in self.fields.items()])
            + ")"
        )


class ActionSpace(Space):
    continuous: Optional[Box] = None
    discrete: Optional[Discrete] = None

    def __init__(self, fields: dict[str, Space]):
        super().__init__()
        self.fields = fields

        for name, space in self.fields.items():
            setattr(self, name, space)

    def sample(self):
        return Action(**{name: space.sample() for name, space in self.fields.items()})

    def contains(self, x: Observation):
        for name, space in self.fields.items():
            if not space.contains(getattr(x, name, None)):
                return False
        return True

    def __repr__(self) -> str:
        return (
            "Action("
            + ", ".join([str(k) + ": " + str(s) for k, s in self.fields.items()])
            + ")"
        )
