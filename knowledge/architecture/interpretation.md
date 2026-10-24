---
title: "interpretation"
type: module
domain: interpretation
parent: "[[wut_murl_craftax]]"
tags: [architecture, interpretation]
created: 2026-10-24
updated: 2026-10-24
related-code:
  - "[[src/wut_murl_craftax/interpretation/__init__.py]]"
  - "[[src/wut_murl_craftax/interpretation/features.py]]"
  - "[[src/wut_murl_craftax/interpretation/dataset.py]]"
  - "[[src/wut_murl_craftax/interpretation/distill.py]]"
---
# interpretation

## Responsibility

- Explain trained policies: observation features, rollout datasets, decision-tree distillation, tree policies and rules ([[2026-10-20-craftax-interpretation-and-ruleset]]).
- `features`: decode symbolic observations of both games into named, human-readable tree inputs.
- `dataset`: label the states a final policy visits with its network's outputs, per seed split.
- `distill`: fit the grid of decision trees to a dataset, score them on held-out states, read the complexity levels off the curve.

## Public surface

- `FEATURES[game] -> list[str]`: column names (111 Craftax, 60 Classic): intrinsics, inventory counts, equipment, potions, flags, facing, creatures, walkable neighbours, then distance and row and column offset signs of the nearest visible target (17 Craftax, 10 Classic).
- `featurize(obs, game, xp=None) -> array`: float32 `(..., len(FEATURES[game]))` from flat observations `(..., D)`; `xp` is numpy or jax.numpy (default: the module of `obs`), so it jits inside closed-loop tree play.
- `featurize` raises `ValueError` when the observation width is not the game's (8268 Craftax, 1345 Classic).
- `CREATURES`: names of the `facing_creature` codes.
- `collect(run, update, seeds, max_states_per_episode=2000, outputs=network_outputs, family=None) -> Dataset`: states of the stochastic policy of one checkpoint (`EvalRun`, ruling R95); `outputs` and `family` are stand-ins of `network_outputs` and `network_family` for toy policies; `None` keeps every state; empty `seeds` raise `ValueError`.
- `Dataset`: `features` (float32), `columns`, `label` (greedy primitive action, int8), `probs` (stochastic distribution), `episode` (seed), `step`, `floor`, `family` (`network_family`), `recurrent`; `save(path)`, `Dataset.load(path)`.
- `RECURRENT`: `ppo-rnn`, `pqn-rnn` and `skills` (the skill network is an LSTM).
- `TRAIN_SEEDS` (30000-30255), `VAL_SEEDS` (= `EVAL_SEEDS`, 10000-10063), `TEST_SEEDS` (20000-20063), `SPLITS`, `MAX_STATES` (2000).
- `dataset_path(budget, method, run, split, root=outputs/interpretation)`: `<root>/<budget>/<method>/<run>/<split>.npz`.
- Columns: `FEATURES[game]`, then `MEMORY_FEATURES` (`prev_action`, -1 at the start; `steps_since_unlock`), then for two-level policies `SKILL_FEATURES` (`skill_<name>` one-hot of the 25 skills, `skill_steps`). Memory proxies are stored for every policy; distillation uses them only when the file's `recurrent` flag is set and reports them apart.
- `fit_grid(train, val, memory=None, skill=None, workers=1) -> list[FittedTree]`: the 12 leaf-grid trees (`LEAF_GRID`, `max_leaf_nodes` 2 to 4096) then the 9 depth-grid trees (`DEPTH_GRID`, `max_depth` 2 to 20 and unlimited), all `criterion="gini"`, `min_samples_leaf=5`, `random_state=0`; `train` and `val` must have the same columns (`ValueError` otherwise).
- `FittedTree`: `kind` (`leaves` or `depth`), `size`, `model` (`DecisionTreeClassifier`), `columns` (the names the model reads, in order), `val_top1` (validation top-1 fidelity, for tree-size choice), `leaves`, `depth`.
- `feature_columns(ds, memory=None, skill=None)`: the columns a tree reads. `memory` defaults to `ds.recurrent` (memory proxies for recurrent policies only); `memory=False` fits a recurrent policy without them to report apart; `memory=True` on a feed-forward policy raises `ValueError`. `skill=False` drops the current-skill columns, default keeps those the dataset has.
- `fidelity(tree, test) -> {"top1", "expected"}`: share of test states where the tree plays the network's greedy action; mean probability the stochastic policy gives the tree's action. Datasets are matched to the tree by column name.
- `fidelity_curve(trees, test) -> list[dict]`: one row per tree (`kind`, `size`, `leaves`, `depth`, `top1`, `expected`, `val_top1`); plain values, ready for `complexity.csv`.
- `complexity(curve) -> {"L80", "L90", "L95", "depth_L90"}`: smallest leaf-grid `max_leaf_nodes` with test top-1 >= 0.8, 0.9, 0.95 (None = never, reported as "> 4096") and the depth of the L90 tree; depth-grid rows are ignored.
- `workers` runs the independent fits in threads (the tree builder releases the GIL); trees are identical for any count. Cost on synthetic noisy data (113 columns, 4 workers): about 17 s for 84,000 train rows, about 250 s for 512,000; the unlimited-depth and depth-20 trees dominate.
- Ordinal codes (a split such as `direction <= 2.5` means "left or right"; rules must translate thresholds into sets):

| Column | Codes |
|---|---|
| `direction` | 1 left, 2 right, 3 up, 4 down (`Action` ids) |
| `facing_creature` | 0 none, 1 melee, 2 passive, 3 ranged, 4 enemy projectile |
| `facing_block` | `BlockType` id; 0 (`INVALID`) is an unlit tile |

## Data flow

- In: flat symbolic observations (`featurize`); runs and network outputs from [[visualisation]] (`EvalRun`, `network_outputs`), final scores from MLflow.
- `collect`: the stored stochastic eval file of the checkpoint when it holds every seed (the validation split), else `rollout_seeds` in batches of 64; then per episode `replay_episode` (observations, dones, unlock events), `outputs` twice over the whole episode (stochastic `probs`, greedy label), uniform subsample, `featurize` on the kept rows only.
- `fit_grid`: one column-restricted float32 matrix per split, shared by all 21 fits; each fit is scored on the validation states at once (`val_top1`).
- Out: `outputs/interpretation/<budget>/` (datasets, trees, rules, `complexity.csv`).

## Invariants

- Policies are chosen by the `phases` tag and `config_ref`, never by experiment name.
- Map decoding is `visualisation.overlay.decode_obs`; flat offsets come from `CRAFTAX_BARS`; no second decoder.
- `featurize` gives identical values under numpy and jitted jax.numpy; numpy runs in chunks of 4096 rows to bound memory.
- With no target in view, its distance is the sentinel (10 Craftax, 8 Classic) and both signs are 0; invisible tiles never count.
- An unlit tile decodes as block 0: never a target, never walkable (unknown is not safe); the game block `DARKNESS` stays walkable.
- `down_ladder` is the `LADDER_DOWN` item; whether it can be used is the `ladder_open` flag.
- Memory proxies and the current skill are not observation features; the dataset adds them.
- Network outputs run over the whole episode before subsampling, so recurrent carries are threaded through every step.
- The subsample of an episode is seeded by its seed: a collection is identical in any seed list, batch or rerun.
- Two-level labels are the primitive action the full policy takes: the skill network's greedy action under the stored running skill, the fixed action on primitive skills (`network_outputs`).
- Rows follow the order of `seeds`, then the step.
- Trees are fitted on train states only; validation states only score (`val_top1`), test states give every reported fidelity and L value.
- Memory proxies reach a tree only through `feature_columns`: never for a feed-forward policy, for a recurrent one by default and reported apart from the fit without them.
- Complexity levels read the leaf grid only; the depth grid is a second view of the same data and never moves L80, L90 or L95.

## Known issues

- None.

## Tests

- `tests/interpretation/test_dataset.py`: seed splits, empty seeds refused, identical repeat collections with family and recurrent flag and outputs equal to `network_outputs` on an untrained Classic PPO checkpoint, steps since the last unlock, uniform subsampling, a recurrent toy whose labels follow its carry, a two-level toy's labels and skill columns, eval-file reuse, save and load.
- `tests/interpretation/test_distill.py`: a synthetic policy with a known 8-leaf rule over 5 features (top-1 1.0 from 8 leaves, L90 at most 8, expected fidelity equal to top-1 for a one-hot policy and by hand for a soft one), grid sizes and tree parameters, validation fidelity, worker counts giving identical trees, memory and skill column selection, a label only a memory proxy explains, matching by column name, `complexity` on hand-made curves.
- `tests/interpretation/test_features.py`: round trips on edited reset states of both games, hand-placed targets and creatures, darkness, ladders, fountain and tree kinds, walkable blockers, Craftax creatures, width check, numpy vs jitted jax, 10,000 observations in whole chunks.
