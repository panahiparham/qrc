import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

import numpy as np
import matplotlib.pyplot as plt

from PyExpUtils.results.LazyCollection import LazyResultCollection
from PyExpPlotting.matplot import save, setDefaultConference
from RlEvaluation.config import data_definition
from RlEvaluation.temporal import TimeSummary, extract_learning_curves, curve_percentile_bootstrap_ci
from RlEvaluation.statistics import Statistic
from RlEvaluation.interpolation import compute_step_return
import RlEvaluation.hypers as Hypers
import RlEvaluation.metrics as Metrics

from experiment.ExperimentModel import ExperimentModel
from experiment.tools import parseCmdLineArgs

setDefaultConference('jmlr')

COLORS = {
    'dqn': 'black',
    'qrc': 'tab:blue',
}
LABELS = {
    'dqn': 'DQN',
    'qrc': 'QRC',
}


if __name__ == '__main__':
    path, should_save, save_type = parseCmdLineArgs()

    results = LazyResultCollection(Model=ExperimentModel, metrics=['steps', 'return'])
    data_definition(
        hyper_cols=results.get_hyperparameter_columns(),
        seed_col='seed',
        time_col='frame',
        environment_col=None,
        algorithm_col=None,
        make_global=True,
    )

    for env, sub_results in results.groupby_directory(level=2):
        fig, ax = plt.subplots(1, 1)
        has_data = False

        for alg_result in sub_results:
            alg = alg_result.sub_path
            if alg not in LABELS:
                continue

            df = alg_result.load()
            if df is None or len(df) == 0:
                print(f'No data for {env}/{alg}')
                continue

            Metrics.add_step_weighted_return(df)

            report = Hypers.select_best_hypers(
                df,
                metric='step_weighted_return',
                prefer=Hypers.Preference.high,
                time_summary=TimeSummary.sum,
                statistic=Statistic.mean,
            )

            print(f'{env} / {alg}: best config = {report.best_configuration}')

            exp = results[env, alg].exp

            xs, ys = extract_learning_curves(
                df,
                hyper_vals=report.best_configuration,
                metric='return',
                interpolation=lambda x, y: compute_step_return(x, y, exp.total_steps),
            )

            xs = np.asarray(xs)[:, ::exp.total_steps // 1000]
            ys = np.asarray(ys)[:, ::exp.total_steps // 1000]

            res = curve_percentile_bootstrap_ci(
                rng=np.random.default_rng(0),
                y=ys,
                statistic=Statistic.mean,
                iterations=1000,
            )

            ax.plot(xs[0], res.sample_stat, label=LABELS[alg], color=COLORS[alg], linewidth=1.5)
            ax.fill_between(xs[0], res.ci[0], res.ci[1], color=COLORS[alg], alpha=0.2)
            has_data = True

        if not has_data:
            plt.close(fig)
            continue

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('Time steps')
        ax.set_xticks([0, 100000])
        ax.set_xticklabels(['0', '100k'])
        ax.set_ylabel('Return')
        ax.set_title(env)
        ax.legend(loc='lower right')

        if should_save:
            save(save_path=f'{path}/plots', plot_name=env, f=fig, height_ratio=0.5)
        else:
            plt.show()

        plt.close(fig)
