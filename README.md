<div align="center">

<div id="user-content-toc" style="margin-bottom: 50px">
  <ul align="center" style="list-style: none;">
    <summary>
      <h1>Decoupling Manifold Modeling and Value Maximization for Offline
Policy Extraction</h1>
    </summary>
  </ul>
</div>

<img src="assets/deflow.png" width="80%">

</div>

## Overview

We introduce DeFlow, a framework that assigns
the responsibility of manifold representation and policy
improvement to distinct components in offline RL.

## Installation

The codebase is built on [FQL](https://github.com/seohongpark/fql), [floq](https://github.com/CMU-AIRe/floq/tree/main). You may look at their repos for installation.

## Usage

The main implementation of DeFlow is in [agents/deflow.py](agents/deflow.py),
and the combination of DeFlow and floq can be found in [agents/deflowvf.py](agents/deflowvf.py)
Here are some example commands (see [the section below](#reproducing-the-main-results) for the complete list):
```bash
# DeFlow on OGBench antmaze-large (offline RL)
python main.py --env_name=antmaze-large-navigate-singletask-task1-v0 --agent.q_agg=min --agent.target_divergence=0.01 --agent.normalize_q_loss=True
# DeFlow on OGBench visual-cube-single (offline RL)
python main.py --env_name=visual-cube-single-play-singletask-task1-v0 --offline_steps=500000 --agent.target_divergence=0.001--agent.encoder=impala_small --p_aug=0.5 --frame_stack=3 --agent.normalize_q_loss=True
# DeFlow on OGBench scene (offline-to-online RL)
python main.py --env_name=scene-play-singletask-v0 --online_steps=1000000 --agent.target_divergence=0.001 --agent.normalize_q_loss=True
```
To fix the muti-step flow at online stage. you can set `--agent.fix_bc_flow_online=True`.

## Tips for hyperparameter tuning

For the target divergence $\delta$, we set it based on the estimated Intrinsic Action Variance (IAV) of the dataset. Specifically, we compute the average variance of actions among $k$-nearest neighbors ($k=5$) for each state in the following tasks and normalize it by the action dimension.

Basically, we observe a strong empirical correlation between the Intrinsic Action Variance (IAV) and the optimal constraint $\delta$:
- *Fine Manipulation Tasks* (e.g., `cube-double-play`, `scene-play`):
    - Estimated IAV: $\approx 0.01$
    - Recommended $\delta$: $0.001$ (i.e., $0.1 \times \text{IAV}$)
- *Navigation/Locomotion Tasks* (e.g., `antsoccer`, `antmaze`):
    - Estimated IAV: $\approx 0.01$
    - Recommended $\delta$: $0.01$ (i.e., $1.0 \times \text{IAV}$)
- *D4RL Benchmarks*: 
    - Recommended $\delta$: $10^{-3}$ (based on data quality)

## Reproducing the main results

We provide the complete list of the **exact command-line flags**
used to produce the main results of DeFlow in the paper.

<details>
<summary><b>Click to expand the full list of commands</b></summary>

### Offline RL

#### DeFlow on state-based OGBench (default tasks)

```bash
# DeFlow on OGBench antmaze-large-navigate-singletask-v0 (=antmaze-large-navigate-singletask-task1-v0)
python main.py --env_name=antmaze-large-navigate-singletask-task1-v0 --agent.q_agg=min --agent.target_divergence=0.01 --agent.normalize_q_loss=True
# DeFlow on OGBench antmaze-giant-navigate-singletask-v0 (=antmaze-giant-navigate-singletask-task1-v0)
python main.py --env_name=antmaze-giant-navigate-singletask-task1-v0 --agent.discount=0.995 --agent.q_agg=min --agent.target_divergence=0.01 --agent.normalize_q_loss=True
# DeFlow on OGBench humanoidmaze-medium-navigate-singletask-v0 (=humanoidmaze-medium-navigate-singletask-task1-v0)
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task1-v0 --agent.discount=0.995 --agent.target_divergence=0.001 --agent.normalize_q_loss=True
# DeFlow on OGBench humanoidmaze-large-navigate-singletask-v0 (=humanoidmaze-large-navigate-singletask-task1-v0)
python main.py --env_name=humanoidmaze-large-navigate-singletask-task1-v0 --agent.discount=0.995 --agent.target_divergence=0.001 --agent.normalize_q_loss=True
# DeFlow on OGBench antsoccer-arena-navigate-singletask-v0 (=antsoccer-arena-navigate-singletask-task4-v0)
python main.py --env_name=antsoccer-arena-navigate-singletask-task4-v0 --agent.discount=0.995 --agent.target_divergence=0.01 --agent.normalize_q_loss=True
# DeFlow on OGBench cube-single-play-singletask-v0 (=cube-single-play-singletask-task2-v0)
python main.py --env_name=cube-single-play-singletask-v0 --agent.target_divergence=0.001 --agent.normalize_q_loss=True
# DeFlow on OGBench cube-double-play-singletask-v0 (=cube-double-play-singletask-task2-v0)
python main.py --env_name=cube-double-play-singletask-v0 --agent.target_divergence=0.001 --agent.normalize_q_loss=True
# DeFlow on OGBench scene-play-singletask-v0 (=scene-play-singletask-task2-v0)
python main.py --env_name=scene-play-singletask-v0 --agent.target_divergence=0.001 --agent.normalize_q_loss=True
# DeFlow on OGBench puzzle-3x3-play-singletask-v0 (=puzzle-3x3-play-singletask-task4-v0)
python main.py --env_name=puzzle-3x3-play-singletask-v0 --agent.target_divergence=0.005 --agent.normalize_q_loss=True
# DeFlow on OGBench puzzle-4x4-play-singletask-v0 (=puzzle-4x4-play-singletask-task4-v0)
python main.py --env_name=puzzle-4x4-play-singletask-v0 --agent.target_divergence=0.005 --agent.normalize_q_loss=True
```
For other tasks, just replace the environment name accordingly.

#### DeFlow on pixel-based OGBench

```bash
# DeFlow on OGBench visual-cube-single-play-singletask-task1-v0
python main.py --env_name=visual-cube-single-play-singletask-task1-v0 --offline_steps=500000 --agent.normalize_q_loss=True --agent.target_divergence=0.001 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3
# DeFlow on OGBench visual-cube-double-play-singletask-task1-v0
python main.py --env_name=visual-cube-double-play-singletask-task1-v0 --offline_steps=500000 --agent.normalize_q_loss=True --agent.target_divergence=0.001 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3
# DeFlow on OGBench visual-scene-play-singletask-task1-v0
python main.py --env_name=visual-scene-play-singletask-task1-v0 --offline_steps=500000 --agent.normalize_q_loss=True --agent.target_divergence=0.001 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3
# DeFlow on OGBench visual-puzzle-3x3-play-singletask-task1-v0
python main.py --env_name=visual-puzzle-3x3-play-singletask-task1-v0 --offline_steps=500000 --agent.normalize_q_loss=True --agent.target_divergence=0.0005 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3
# DeFlow on OGBench visual-puzzle-4x4-play-singletask-task1-v0
python main.py --env_name=visual-puzzle-4x4-play-singletask-task1-v0 --offline_steps=500000 --agent.normalize_q_loss=True --agent.target_divergence=0.0005 --agent.encoder=impala_small --p_aug=0.5 --frame_stack=3
```

#### DeFlow on D4RL

```bash
# DeFlow on D4RL antmaze-umaze-v2
python main.py --env_name=antmaze-umaze-v2 --offline_steps=500000 --agent.normalize_q_loss=True --agent.target_divergence=0.015
# DeFlow on D4RL antmaze-umaze-diverse-v2
python main.py --env_name=antmaze-umaze-diverse-v2 --offline_steps=500000 --agent.normalize_q_loss=True --agent.target_divergence=0.015
# DeFlow on D4RL antmaze-medium-play-v2
python main.py --env_name=antmaze-medium-play-v2 --offline_steps=500000 --agent.normalize_q_loss=True --agent.target_divergence=0.015
# DeFlow on D4RL antmaze-medium-diverse-v2
python main.py --env_name=antmaze-medium-diverse-v2 --offline_steps=500000 --agent.normalize_q_loss=True --agent.target_divergence=0.015
# DeFlow on D4RL antmaze-large-play-v2
python main.py --env_name=antmaze-large-play-v2 --offline_steps=500000 --agent.normalize_q_loss=True --agent.target_divergence=0.015
# DeFlow on D4RL antmaze-large-diverse-v2
python main.py --env_name=antmaze-large-diverse-v2 --offline_steps=500000 --agent.normalize_q_loss=True --agent.target_divergence=0.015
# DeFlow on D4RL pen-human-v1
python main.py --env_name=pen-human-v1 --offline_steps=500000 --agent.q_agg=min --agent.normalize_q_loss=True --agent.target_divergence=0.01
# DeFlow on D4RL pen-cloned-v1
python main.py --env_name=pen-cloned-v1 --offline_steps=500000 --agent.q_agg=min --agent.normalize_q_loss=True --agent.target_divergence=0.01
# DeFlow on D4RL pen-expert-v1
python main.py --env_name=pen-expert-v1 --offline_steps=500000 --agent.q_agg=min --agent.normalize_q_loss=True --agent.target_divergence=0.01
# DeFlow on D4RL door-human-v1
python main.py --env_name=door-human-v1 --offline_steps=500000 --agent.q_agg=min --agent.normalize_q_loss=True --agent.target_divergence=0.001
# DeFlow on D4RL door-cloned-v1
python main.py --env_name=door-cloned-v1 --offline_steps=500000 --agent.q_agg=min --agent.normalize_q_loss=True --agent.target_divergence=0.001
# DeFlow on D4RL door-expert-v1
python main.py --env_name=door-expert-v1 --offline_steps=500000 --agent.q_agg=min --agent.normalize_q_loss=True --agent.target_divergence=0.001
# DeFlow on D4RL hammer-human-v1
python main.py --env_name=hammer-human-v1 --offline_steps=500000 --agent.q_agg=min --agent.normalize_q_loss=True --agent.target_divergence=0.001
# DeFlow on D4RL hammer-cloned-v1
python main.py --env_name=hammer-cloned-v1 --offline_steps=500000 --agent.q_agg=min --agent.normalize_q_loss=True --agent.target_divergence=0.001
# DeFlow on D4RL hammer-expert-v1
python main.py --env_name=hammer-expert-v1 --offline_steps=500000 --agent.q_agg=min --agent.normalize_q_loss=True --agent.target_divergence=0.001
# DeFlow on D4RL relocate-human-v1
python main.py --env_name=relocate-human-v1 --offline_steps=500000 --agent.q_agg=min --agent.normalize_q_loss=True --agent.target_divergence=0.001
# DeFlow on D4RL relocate-cloned-v1
python main.py --env_name=relocate-cloned-v1 --offline_steps=500000 --agent.q_agg=min --agent.normalize_q_loss=True --agent.target_divergence=0.001
# DeFlow on D4RL relocate-expert-v1
python main.py --env_name=relocate-expert-v1 --offline_steps=500000 --agent.q_agg=min --agent.normalize_q_loss=True --agent.target_divergence=0.001
```

### Offline-to-online RL

Just add `--online_steps=1000000` to the offline RL commands above and change environment names accordingly if needed.
</details>

## Acknowledgments

This codebase is built on top of [FQL](https://github.com/seohongpark/fql) and [floq](https://github.com/CMU-AIRe/floq/tree/main).
Thank the authors for their great works.
