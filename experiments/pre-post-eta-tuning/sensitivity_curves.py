import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
import polars as pl
import matplotlib.pyplot as plt

from PyExpUtils.results.LazyCollection import LazyResultCollection
from PyExpPlotting.matplot import save, setDefaultConference
from rlevaluation.config import data_definition
from rlevaluation.temporal import TimeSummary, extract_learning_curves, curve_percentile_bootstrap_ci
from rlevaluation.statistics import Statistic
import rlevaluation.hypers as Hypers

from experiment.ExperimentModel import ExperimentModel

setDefaultConference('jmlr')

PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots')

COLORS = {
    'dqn': 'black',
    'qrc': 'tab:blue',
    'qrc-target': 'tab:purple',
}
LABELS = {
    'dqn': 'DQN',
    'qrc': 'QRC',
    'qrc-target': 'QRC + Target Net',
}

results = LazyResultCollection(Model=ExperimentModel, metrics=['steps', 'return'])
hyper_cols = results.get_hyperparameter_columns()
data_definition(hyper_cols=hyper_cols, make_global=True)

for env, sub_results in results.groupby_directory(level=2):
    fig, ax = plt.subplots(1, 1)
    has_data = False

    for alg_result in sub_results:
        alg = alg_result.sub_path
        if alg not in LABELS:
            continue

        pdf = alg_result.load()
        if pdf is None or len(pdf) == 0:
            print(f'No data for {env}/{alg}')
            continue
        df = pl.from_pandas(pdf)

        group_cols = ['seed'] + [c for c in hyper_cols if c in df.columns]
        df = df.with_columns(
            (pl.col('return') * (pl.col('steps') / pl.col('frame').max().over(group_cols)))
            .alias('step_weighted_return')
        )

        perfs = []

        for (eta,), sub_df in df.group_by('optimizer.learning_rate'):
            report = Hypers.select_best_hypers(
                sub_df,
                metric='step_weighted_return',
                prefer=Hypers.Preference.high,
                time_summary=TimeSummary.sum,
                statistic=Statistic.mean,
            )

            # print(f'{env} / {alg}: best config = {report.best_configuration} / eta: {eta}')

            exp = results[env, alg].exp

            _, ys = extract_learning_curves(
                df,
                hyper_vals=report.best_configuration,
                metric='return',
                # interpolation=lambda x, y: compute_step_return(x, y, exp.total_steps),
            )

            ys = [y.mean() for y in ys]
            ys = np.asarray(ys)
            perfs.append((eta, np.asarray(ys)))


        # sort by learning rates
        perfs = sorted(perfs, key=lambda x: x[0])

        etas, ys = zip(*perfs)

        etas = np.asarray(etas)
        # find the minimum length of the arrays in ys
        min_len = min(len(y) for y in ys)
        # truncate all arrays in ys to the minimum length
        ys = [y[:min_len] for y in ys]
        ys = np.asarray(ys).T

        print(etas.shape)
        print(ys.shape)

        res = curve_percentile_bootstrap_ci(
            rng=np.random.default_rng(0),
            y=ys,
            statistic=Statistic.mean,
            iterations=1000,
        )

        ax.plot(etas, res.sample_stat, label=LABELS[alg], color=COLORS[alg], linewidth=1.5)
        ax.fill_between(etas, res.ci[0], res.ci[1], color=COLORS[alg], alpha=0.2)
        has_data = True

        if not has_data:
            plt.close(fig)
            continue

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('H-head Learning Rate Multiplier (eta)')
        # ax.set_xticks([2 ** -i for i in range(14, 3, -1)])
        ax.set_xscale('log', base=2)
        ax.set_ylabel('Average Lifetime Return')
        ax.set_title(env)
        ax.legend(loc='upper right')

        save(save_path=PLOTS_DIR, plot_name=env, f=fig, height_ratio=0.5)

        plt.close(fig)
