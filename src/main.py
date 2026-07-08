import os
import sys
sys.path.append(os.getcwd())

import time
import logging
import argparse
import numpy as np
from rlglue import RlGlue
from experiment import ExperimentModel
from problems.registry import getProblem
from PyExpUtils.results.sqlite import saveCollector
from PyExpUtils.collection.Collector import Collector
from PyExpUtils.collection.Sampler import Ignore, MovingAverage, Subsample, Identity
from PyExpUtils.collection.utils import Pipe

# ------------------
# -- Command Args --
# ------------------
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exp', type=str, required=True)
parser.add_argument('-i', '--idxs', nargs='+', type=int, required=True)
parser.add_argument('--save_path', type=str, default='./')
parser.add_argument('--silent', action='store_true', default=False)

args = parser.parse_args()

# ---------------------------
# -- Library Configuration --
# ---------------------------
import jax
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('jax').setLevel(logging.WARNING)
logger = logging.getLogger('exp')
if not args.silent:
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)

# ----------------------
# -- Experiment Def'n --
# ----------------------
exp = ExperimentModel.load(args.exp)
indices = args.idxs

Problem = getProblem(exp.problem)
for idx in indices:
    run = exp.getRun(idx)
    np.random.seed(run + exp.seed_offset)

    n = exp.total_steps // 100
    collector = Collector(
        config={
            'return': Identity(),
            'episode': Identity(),
            'steps': Identity(),
            'v_loss': Pipe(MovingAverage(0.999), Subsample(n)),
            'h_loss': Pipe(MovingAverage(0.999), Subsample(n)),
        },
        default=Ignore(),
    )
    collector.setIdx(idx)

    problem = Problem(exp, idx, collector)
    agent = problem.getAgent()
    env = problem.getEnvironment()

    glue = RlGlue(agent, env)

    episode = 0
    start_time = time.time()
    glue.start()

    for step in range(exp.total_steps):
        collector.next_frame()
        interaction = glue.step()

        if (interaction.term or interaction.trunc) or \
           (exp.episode_cutoff > -1 and glue.num_steps >= exp.episode_cutoff):
            agent.cleanup()
            collector.collect('return', glue.total_reward)
            collector.collect('episode', episode)
            collector.collect('steps', glue.num_steps)
            episode += 1

            fps = step / (time.time() - start_time) if step > 0 else 0
            avg_time = 1000 * (time.time() - start_time) / (step + 1)
            logger.debug(f'ep={episode} step={step} return={glue.total_reward:.1f} {avg_time:.3}ms/step {int(fps)}fps')

            glue.start()

    collector.reset()

    # ------------
    # -- Saving --
    # ------------
    saveCollector(exp, collector, base=args.save_path)
