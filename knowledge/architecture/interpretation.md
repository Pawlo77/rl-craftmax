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
  - "[[src/wut_murl_craftax/interpretation/tree_policy.py]]"
---
# interpretation

## Responsibility

- Explain trained policies: observation features, rollout datasets, decision-tree distillation, tree policies and rules ([[2026-10-20-craftax-interpretation-and-ruleset]]).
- `features`: decode symbolic observations of both games into named, human-readable tree inputs.
- `dataset`: label the states a final policy visits with its network's outputs, per seed split.
- `distill`: fit the grid of decision trees to a dataset, score them on held-out states, read the complexity levels off the curve.
- `tree_policy`: play a fitted tree as a policy and score it in closed loop against its network.

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
- `FittedTree`: `kind` (`leaves` or `depth`), `size`, `model` (`DecisionTreeClassifier`), `columns` (the names the model reads, in order), `val_top1` (validation top-1 fidelity; reported, no size is selected by it), `leaves`, `depth`.
- `feature_columns(ds, memory=None, skill=None)`: the columns a tree reads. `memory` defaults to `ds.recurrent` (memory proxies for recurrent policies only); `memory=False` fits a recurrent policy without them to report apart; `memory=True` on a feed-forward policy raises `ValueError`. `skill=False` drops the current-skill columns, default keeps those the dataset has; `skill=True` without skill columns raises `ValueError`.
- `fidelity(tree, test) -> {"top1", "expected"}`: share of test states where the tree plays the network's greedy action; mean probability the stochastic policy gives the tree's action. Datasets are matched to the tree by column name.
- `fidelity_curve(trees, test) -> list[dict]`: one row per tree (`kind`, `size`, `leaves`, `depth`, `top1`, `expected`, `val_top1`, `memory`, `skill`: whether the tree reads memory-proxy or current-skill columns, so the fits of one recurrent policy stay distinguishable); plain values, ready for `complexity.csv`.
- `complexity(curve) -> {"L80", "L90", "L95", "depth_L90"}`: smallest leaf-grid `max_leaf_nodes` with test top-1 >= 0.8, 0.9, 0.95 (None = never, reported as "> 4096") and the depth of the L90 tree; depth-grid rows are ignored.
- `workers` runs the independent fits in threads (the tree builder releases the GIL); trees are identical for any count.
- `TreePolicy(tree, game, top_level=None)`: a `FittedTree` as a deterministic policy under both protocols.
  - `init_carry(n)` and `step(params, carry, obs, key, greedy, reward=None, done=None)`: `training.interface.Policy`, what `rollout_seeds` drives.
  - `__call__(state, rng)` and `name = "tree"`: `visualisation.rollout.Policy`, for `play`.
  - `extras(carry)`: the memory proxies it read (`prev_action`, `steps_since_unlock`) and the top level's extras.
  - `ValueError`: a column it cannot compute (skill columns without `top_level`); `steps_since_unlock` without `reward` in `step`; `__call__` with a top level, whose params arrive through `step` only.
- `TreeCarry`: `memory` (proxies, last action, last health, started) and `top` (the top level's carry, `()` without one).
- `tree_rollouts(tree, run, update, seeds) -> dict`: eval-file arrays (`rollout_seeds`) of the tree played greedily with the eval keys; a run whose policy has `extras` is two-level and becomes the top level.
- `closed_loop(tree, run, update, seeds=TEST_SEEDS, network=None) -> ClosedLoop`: per-seed greedy scores (% of max) of the tree and of the network on the same seeds; `network` reuses the network's scores of an earlier call. `ClosedLoop.fidelity` = mean tree score / mean network score (NaN when the network's is <= 0).
- `CLOSED_LOOP_LEAVES` (256): the one leaf-grid size played in closed loop (preflight I4).
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
- `closed_loop`: the network's greedy `rollout_seeds`, then the same with the run's policy swapped for a `TreePolicy` (`dataclasses.replace`), so both play the eval keys; returns from the events (`eval_returns`).
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
- A tree policy computes its columns as `collect` does: `featurize` (jax.numpy), `prev_action` the action played one step earlier (-1 at a start), `steps_since_unlock` reset by an unlock; `step` detects an unlock as reward minus 0.1 x the health change above 0.5 (the smallest unlock weight is 1), `__call__` from the state's achievements.
- Tree splits run in float32 with each threshold rounded down to float32, so a JAX tree plays exactly what scikit-learn predicts.
- Two-level trees replace only the skill network: the top level gets params, key, greedy, reward and done and keeps choosing and ending skills; primitive skills keep their fixed action; an `SMDPPolicy` carry gets the played action as its last action.
- Complexity levels read the leaf grid only; the depth grid is a second view of the same data and never moves L80, L90 or L95.

## Known issues

- None.

## Tests

- `tests/interpretation/test_dataset.py`: seed splits, empty seeds refused, identical repeat collections with family and recurrent flag and outputs equal to `network_outputs` on an untrained Classic PPO checkpoint, steps since the last unlock, uniform subsampling, a recurrent toy whose labels follow its carry, a two-level toy's labels and skill columns, eval-file reuse, save and load.
- `tests/interpretation/test_distill.py`: a synthetic policy with a known 8-leaf rule over 5 features (top-1 1.0 from 8 leaves, L90 at most 8, expected fidelity equal to top-1 for a one-hot policy and by hand for a soft one), grid sizes and tree parameters, validation fidelity, worker counts giving identical trees (features, thresholds, node sizes), memory and skill column selection (a tree splitting on a skill column, curve flags), a label only a memory proxy explains, matching by column name, `complexity` on hand-made curves.
- `tests/interpretation/test_tree_policy.py`: actions on replayed states and 100 Classic steps of a 4-leaf tree through `play`, memory proxies online equal to `collect`'s offline ones with actions equal to scikit-learn's on those rows, both protocols playing the same episodes, errors for missing reward and uncomputable columns, a two-level toy keeping its skills, primitive actions and the reward and done it is handed, an `smdp-skills` checkpoint reading the played action, closed-loop fidelity 1 for a tree equal to its toy network, float32 splits at every threshold.
- `tests/interpretation/test_features.py`: round trips on edited reset states of both games, hand-placed targets and creatures, darkness, ladders, fountain and tree kinds, walkable blockers, Craftax creatures, width check, numpy vs jitted jax, 10,000 observations in whole chunks.
