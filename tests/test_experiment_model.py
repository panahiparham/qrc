import json
import pytest
from pathlib import Path
import experiment.ExperimentModel as ExperimentModule


MINIMAL_EXP = {
    "agent": "DQN",
    "problem": "Acrobot",
    "total_steps": 1000,
    "episode_cutoff": 500,
    "metaParameters": {
        "epsilon": 0.1,
        "buffer_type": "iid",
        "buffer_size": 100,
        "batch": 8,
        "target_refresh": 4,
        "optimizer": {
            "name": "ADAM",
            "learning_rate": [0.001, 0.01],
        },
        "representation": {"type": "TwoLayerRelu", "hidden": 8},
        "gamma": 0.99,
        "n_step": 1,
    },
}


@pytest.fixture
def exp_file(tmp_path):
    path = tmp_path / "test_exp.json"
    path.write_text(json.dumps(MINIMAL_EXP))
    return str(path)


def test_load_from_json(exp_file):
    exp = ExperimentModule.load(exp_file)
    assert exp.agent == 'DQN'
    assert exp.problem == 'Acrobot'
    assert exp.total_steps == 1000
    assert exp.episode_cutoff == 500


def test_num_permutations(exp_file):
    exp = ExperimentModule.load(exp_file)
    # 2 learning rates × 1 run = 2 permutations
    assert exp.numPermutations() == 2


def test_get_hypers_returns_dict(exp_file):
    exp = ExperimentModule.load(exp_file)
    hypers = exp.get_hypers(0)
    assert isinstance(hypers, dict)
    assert 'epsilon' in hypers
    assert 'buffer_type' in hypers


def test_get_hypers_lr_sweep(exp_file):
    exp = ExperimentModule.load(exp_file)
    lr0 = exp.get_hypers(0)['optimizer']['learning_rate']
    lr1 = exp.get_hypers(1)['optimizer']['learning_rate']
    assert lr0 != lr1
    assert set([lr0, lr1]) == {0.001, 0.01}


def test_get_run(exp_file):
    exp = ExperimentModule.load(exp_file)
    assert isinstance(exp.getRun(0), int)
