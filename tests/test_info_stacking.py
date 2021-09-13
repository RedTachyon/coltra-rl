import numpy as np

from coltra.utils import np_float
from coltra.envs.subproc_vec_env import _flatten_info


def test_flatten_info():
    # I had some problems with this, so an explicit test
    infos = [
        {"foo": "asdf", "m_metric": np_float(1.0)},
        {"bar": 1, "m_metric": np_float(2.0)},
        {},
        {"foo": "potato", "bar": "saf"},
    ]

    info = _flatten_info(infos)

    assert isinstance(info["m_metric"], np.ndarray)
    assert info["m_metric"].shape == (2,)
    assert all(info["m_metric"] == np.array([1, 2], dtype=np.float32))
    assert info["foo"] == ["asdf", None, None, "potato"]
    assert info["bar"] == [None, 1, None, "saf"]
