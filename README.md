# qrc

This repo implements two variants of the QRC algorithm (https://proceedings.mlr.press/v119/ghiassian20a.html, https://www.jmlr.org/papers/v23/21-037.html) and runs them on three classic control environments. There is also a DQN implementation for comparison.

The two variants of QRC differ in the way the h-head regularization is applied. In `src/algorithms/nn/QRC.py` the regularization is applied as an L2 term in the loss but in `src/algorithms/nn/QRCPostAdam.py` it is applied as weight decay after the Adam update calculates the weight change.

## usage

Assuming a monolithic cpu system (like your laptop) with uv installed you can install dependencies with:

```bash
uv sync
```

Then you can run a batch of experiments with:

```bash
uv run scripts/local.py --runs 3 --cpus 4 -e experiments/pre-post-adam-qrc/**/*.json
```

Finally plot the results with:

```bash
uv run experiments/pre-post-adam-qrc/plot_learning_curves.py
```

The `experiments/pre-post-sgd-qrc` experiment is identical except the agents use SGD instead of Adam; run and plot it by swapping the experiment directory in the commands above.
