---
title: "interpretation"
type: module
domain: interpretation
parent: "[[wut_murl_craftax]]"
tags: [architecture, interpretation]
created: 2026-10-24
updated: 2026-10-29
related-code:
  - "[[src/wut_murl_craftax/interpretation/__init__.py]]"
  - "[[src/wut_murl_craftax/interpretation/features.py]]"
  - "[[src/wut_murl_craftax/interpretation/dataset.py]]"
  - "[[src/wut_murl_craftax/interpretation/distill.py]]"
  - "[[src/wut_murl_craftax/interpretation/tree_policy.py]]"
  - "[[src/wut_murl_craftax/interpretation/rules.py]]"
  - "[[src/wut_murl_craftax/interpretation/run.py]]"
---
# interpretation

## Responsibility

- Explain trained policies: observation features, rollout datasets, decision-tree distillation, tree policies and rules ([[2026-10-20-craftax-interpretation-and-ruleset]]).
- `features`: decode symbolic observations of both games into named, human-readable tree inputs.
- `dataset`: label the states a final policy visits with its network's outputs, per seed split.
- `distill`: fit the grid of decision trees to a dataset, score them on held-out states, read the complexity levels off the curve.
- `tree_policy`: play a fitted tree as a policy and score it in closed loop against its network.
- `rules`: read a fitted tree as rules in plain words, rank its features, break down where it fails, and report the top-level rules of the skill methods from their greedy table.
- `run`: the CLI that distils every final policy, sums each method and budget up (`complexity.csv`, the fidelity figure) and logs each method to MLflow.

## Public surface

- `FEATURES[game] -> list[str]`: column names (111 Craftax, 60 Classic): intrinsics, inventory counts, equipment, potions, flags, facing, creatures, walkable neighbours, then distance and row and column offset signs of the nearest visible target (17 Craftax, 10 Classic).
- `featurize(obs, game, xp=None) -> array`: float32 `(..., len(FEATURES[game]))` from flat observations `(..., D)`; `xp` is numpy or jax.numpy (default: the module of `obs`), so it jits inside closed-loop tree play.
- `featurize` raises `ValueError` when the observation width is not the game's (8268 Craftax, 1345 Classic).
- `CREATURES`: names of the `facing_creature` codes.
- `collect(run, update, seeds, max_states_per_episode=2000, outputs=network_outputs, family=None) -> Dataset`: states of the stochastic policy of one checkpoint (`EvalRun`, ruling R95); `outputs` and `family` are stand-ins of `network_outputs` and `network_family` for toy policies; `None` keeps every state; empty `seeds` raise `ValueError`.
- `relabel(run, update, arrays, max_states_per_episode=2000, outputs=network_outputs, family=None) -> Dataset`: the states of eval-file arrays of any policy (a tree's `tree_rollouts`, for DAgger) labelled by the network of `update`, as `collect` labels its own; the stochastic arrays of `update` give `collect`'s rows.
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
  - `ValueError`: a column it cannot compute (skill columns without `top_level`); a missing `reward` in `step` when the tree reads `steps_since_unlock` or has a top level; `__call__` with a top level, whose params arrive through `step` only.
- `TreeCarry`: `memory` (proxies, last action, last health, started) and `top` (the top level's carry, `()` without one).
- `tree_rollouts(tree, run, update, seeds) -> dict`: eval-file arrays (`rollout_seeds`) of the tree played greedily with the eval keys; a run whose policy has `extras` is two-level and becomes the top level. The online `steps_since_unlock` is checked against the count from `events/unlocked`: `RuntimeError` on a mismatch when the tree reads it, a warning otherwise.
- `closed_loop(tree, run, update, seeds=TEST_SEEDS, network=None) -> ClosedLoop`: per-seed greedy scores (% of max) of the tree and of the network on the same seeds; `network` reuses the network's scores of an earlier call (`ValueError` unless one per seed). `ClosedLoop.fidelity` = mean tree score / mean network score (NaN when the network's is <= 0).
- `CLOSED_LOOP_LEAVES` (256): the one leaf-grid size played in closed loop (preflight I4).
- `extract_rules(tree) -> list[Rule]`: one `Rule(conditions, action, samples, share, purity)` per leaf, left to right; `Condition(feature, lo, hi)` is an inclusive interval, the conditions of a path merged per feature. Integer columns cut at `floor(t)`; the one fractional column, `light_level`, keeps the cut as the float32 values either side of the threshold and prints them as the shortest float32 text, so the printed rule selects exactly what the tree sends to the leaf. `action` is the primitive action id (a skill name for a key tree).
- `rules_text(tree, game=None) -> str`: one numbered line per leaf, `IF <conditions> THEN <ACTION> (<share> of states, purity <p>)`; `game` defaults to Classic when every observation column is one of its 60, else Craftax. `condition_text(condition, game="craftax")` prints one condition in decoded units; a memory-proxy condition ends in `[memory proxy]`, so recurrent rules show them apart.
- `paths_to(tree, action) -> list[Rule]`: every leaf that plays an action id or name (`"DO"`, any case); `action_id(action)` resolves it (`ValueError` on an unknown name).
- `importance(tree, test, repeats=5, seed=0) -> list[dict]`: permutation importance on test states, one row per tree column (`feature`, `group` = observation, memory or skill, `importance` = drop in top-1 fidelity, `std`), largest first; columns the tree never splits on are 0 without a shuffle. `ValueError` without test states.
- `failure_breakdown(tree, test) -> dict`: `top1`, `groups` (`ACTION_GROUPS` keys), `confusion[i, j]` (network action in group i, tree action in group j) and `by_floor` (`floor`, `states`, `top1`, floors ascending).
- `ACTION_GROUPS`: group -> action names (noop, move, do, rest, place, craft, ladder, combat, potion, upgrade), a partition of the `Action` enum (Classic shares ids 0-16).
- `top_level_rules(table, update=None, tried=False, top=10) -> TopLevel`: `table` is the frame or parquet file of `eval/smdp_greedy_table.parquet` (SMDP and Dyna-Q); rows are filtered to `update` (default the largest: the table holds every checkpoint; pass it when the final checkpoint stored no keys). Fields: `update`, `n_rules` (visited keys), `skills` (distinct greedy skills, most frequent first), `counts`, `levels` (`complexity` of the key tree: L80, L90, L95 over keys, `depth_L90`), `curve`, `rules` and `tree_text` (the L90 tree, else the `KEY_LEAVES` = 16-leaf one), `top` (the `top` highest-value keys: `key`, `skill`, `value`, `text`), `n_optimistic` and `optimistic_share` (keys whose `greedy_skill` differs from `greedy_tried_skill`, counted on all keys of the checkpoint, either view). **The tried view is the informative one**: at `q_init` 10 most final keys were never updated off it, so the default argmax is a tie broken by skill order (on all 20 final-1m tables 96-100% of keys, `top_level.json`), and the default's top-value rules are ties at 10.00. Report the tried view as the main one and the default beside it, with `optimistic_share`. `tried=True` uses `greedy_tried_skill` and `greedy_tried_value` and drops keys without a tried skill. `ValueError` for a table without rows; an `update` without rows gives an empty `TopLevel`.
- CLI `python -m wut_murl_craftax.interpretation.run --phases final-1m combo-1m --method all --out outputs/interpretation --tracking-uri sqlite:///<db>?timeout=300 [--dagger] [--refs tuned] [--workers 4] [--figures outputs/results/figures] [--skip-missing] [--force] [--no-log] [--max-seeds N]`: runs from the export's `runs` table (`analysis.load.load_phases`, tag `phases`, R77), `config_ref` `tuned` by default; run folders from `analysis.cases.run_dirs` (one short read per method); `--max-seeds` keeps the first N seeds of each split (tests and smokes only). Exit 1 after the rest when a policy failed.
- `distil_policy(run, row, folder, dagger=False, workers=1, seeds=SPLITS) -> dict`: one run at its final checkpoint (`run.updates[-1]`, passed on as `update=`). Per-policy files in `<out>/<budget>/<method>/<run_id>/`: `train.npz`, `val.npz`, `test.npz` (and `dagger.npz`); per fit `curve_<v>.csv`, `closed_loop_<v>.csv` (`eval_seed`, `tree`, `network`), `importance_<v>.csv` and `failure_<v>.json` of the 256-leaf tree, `rules_16_<v>.txt`, `tree_16_<v>.pkl`, `tree_256_<v>.pkl` (pickled `FittedTree`; the other 19 fits live on as curve rows), `fit_<v>.json` (levels, fidelity at 16, 64, 256, expected fidelity and closed-loop ratio at 256, network scores); skill methods `top_level.json` and `top_level_rules.txt`; `complexity.json` last.
- Fits `v` (`VARIANTS`): `base` (`feature_columns` defaults: memory proxies for recurrent policies, skill columns for two-level ones), `no-memory` (recurrent policies only; a feed-forward policy's no-memory row repeats its base fit), `dagger` (`--dagger`: the base 256-leaf tree played on the training seeds, `relabel`, refit on the union).
- Splits: `CAPS` states per episode, train 2000 (plan), validation 2000 (it only scores `val_top1`), test 10,000 (keeps whole 1M episodes, bounds a 1B one).
- Method folder `<out>/<budget>/<method>/`: `curve.csv`, `closed_loop.csv`, `importance.csv` (every policy and fit, identity columns in front), `complexity.json` (`median_run`, every policy summary), `rules_16.txt` (the seed with the lower-median final score, one section per fit), `fidelity_vs_leaves.png` (a line per seed) and its `captions.csv`, skill methods `top_level.json` (every seed, tried view main, default beside it with `optimistic_share`) and the median seed's `top_level_rules.txt`; logged as run `distill-<budget>-<method>` of experiment `analysis` (tags `analysis/kind=distill`, `analysis/phases`, `analysis/budget`, `analysis/method`) without the `*.npz` datasets; `.logged` holds the run id.
- Budget folder: `complexity.csv` (`COLUMNS`: contracts section 3 plus `top_optimistic_share`, `l90_censored` and the R106 columns `majority`, `gain_16/64/256`, `peak_val_leaves`, `group_<action group>` and `spearman_score_fid256`/`spearman_fid256_p`, all read from the policy folders by `policy_extras`: `test.npz` labels, `curve_<variant>.csv` `val_top1`, `failure_<variant>.json`), a sibling `policies.csv` (one row per policy and variant with the same numbers), one row per (method, config_ref, variant) over every policy summary under the folder; `fidelity_vs_leaves.png` and `captions.csv` (id `FID`) (PNG only; an old `.pdf` is deleted; report theme, column width, size class curve, no legend, title or label on the image: the method legend figure gives the colours, the caption the 0.9 level, median base curve per unit), copied to `--figures/<budget>/` with its caption row.
- `censored_median(levels)`: lower median with None as inf, so it is a grid size and inf exactly when more than half the seeds never reach the level; `level_text` prints inf as `> 4096`; `spearman(scores, levels)`: rho and p with None ranked above 4096, NaN under 3 seeds, under `latex.MIN_UNCENSORED` = 3 reached levels or for constant scores or levels (no scipy warning); `uncensored_median(values)`: the L90 depth median, NaN under 3 reached seeds; `complexity_table(summaries, budget)`.
- Ordinal codes (a split such as `direction <= 2.5` means "left or right"; rules must translate thresholds into sets):

| Column | Codes |
|---|---|
| `direction` | 1 left, 2 right, 3 up, 4 down (`Action` ids) |
| `facing_creature` | 0 none, 1 melee, 2 passive, 3 ranged, 4 enemy projectile |
| `facing_block` | `BlockType` id; 0 (`INVALID`) is an unlit tile, printed "unlit/invalid" |
| `prev_action` | -1 start, else `Action` id |
| `<target>_row_sign`, `hostile_row_sign` | -1 above, 0 level or nothing in view, 1 below |
| `<target>_col_sign`, `hostile_col_sign` | -1 left, 0 level or nothing in view, 1 right |

`rules` prints these by name, never as a threshold: the codes an interval selects, or "not" plus the other codes when those are fewer ("facing_creature is not none"); an interval that selects no code raises `ValueError`.

## Data flow

- In: flat symbolic observations (`featurize`); runs and network outputs from [[visualisation]] (`EvalRun`, `network_outputs`), final scores from MLflow.
- `collect`: the stored stochastic eval file of the checkpoint when it holds every seed (the validation split), else `rollout_seeds` in batches of 64; then per episode `replay_episode` (observations, dones, unlock events), `outputs` twice over the whole episode (stochastic `probs`, greedy label), uniform subsample, `featurize` on the kept rows only.
- `fit_grid`: one column-restricted float32 matrix per split, shared by all 21 fits; each fit is scored on the validation states at once (`val_top1`).
- `closed_loop`: the network's greedy `rollout_seeds`, then the same with the run's policy swapped for a `TreePolicy` (`dataclasses.replace`), so both play the eval keys; returns from the events (`eval_returns`).
- `rules`: a tree's leaves are walked with scikit-learn's arrays (`tree_`), no re-prediction; `importance` and `failure_breakdown` predict on the test matrix in the tree's columns; `top_level_rules` fits its own trees on the final checkpoint's keys, each key one row.
- `run`: export `runs` rows, then per policy `EvalRun.load`, `collect` per split (or the saved file), `fit_grid` per fit, `fidelity_curve`, `complexity`, `closed_loop` of the 256-leaf tree (the network's greedy scores once, reused by the other fits), `rules_text`, `importance`, `failure_breakdown`, `top_level_rules` (both views, `update=` the final checkpoint); then the method and budget summaries and the MLflow logging.
- Out: `outputs/interpretation/<budget>/` (datasets, trees, rules, `complexity.csv`), the figure in `outputs/results/figures/`.

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

- The leaves of a rules tree partition the states: the merged conditions of one leaf select exactly the rows the tree sends to it.
- Rule text prints decoded units only: counts and levels as integers, flags as their name, ordinal codes as names; memory proxies carry a suffix and importance rows a `group`, so a recurrent policy's memory use is reported apart from its observation features.
- The key tree has no held-out keys: every visited key is a rule of the top level, so its fidelity is on the keys themselves (`min_samples_leaf=1`) and a table with one greedy skill gets L90 = 2 (the grid's smallest size) from a one-leaf tree.
- The greedy skill of a key is the unmasked argmax of its row, so an untried optimistic entry can win; `tried=True` is the view without them.
- Only the final checkpoint is read: the greedy table holds every checkpoint, so rows are always filtered to one `update`.
- `run` distils at `run.updates[-1]` and passes that update to every reader; a greedy table without keys at it fails the policy (no silent fallback to an earlier checkpoint).
- `run` is resumable per policy: datasets, each fit and the top level are written once, atomically (`analysis.load.atomic_path`: a temporary file of the process beside the target; an error removes it, a kill leaves it), and `complexity.json` last; a policy with it (and its `dagger` fit under `--dagger`) is skipped; `--force` empties the policy folder first. The summary records `split_seeds` (seeds per split): one written for other counts (a `--max-seeds` smoke) is not skipped, its folder is emptied and the policy redone; a summary without the key counts as finished.
- Censored levels: L-values that are None stay None in the policy files, inf in `complexity.csv`, `> 4096` in text; key-tree levels go only to `top_l90`, never the L90 columns or the Spearman.
- `run` never writes the training store: datasets come from fresh rollouts or stored eval files, and only `distill-*` runs of `analysis` are written (none with `--no-log`).

## Known issues

- `depth_at_l90_median` is the median over the seeds that reach L90 only, NaN under 3 of them (`uncensored_median`); a `complexity.csv` written before that rule has 1 for 1M `smdp-skills` (one seed), and `latex` writes no macro for it.
- `spearman_score_l90` ranks censored L90 above 4096: at 1M 89 of 90 base policies (267 of 270 fits) are censored, so the statistic rests on one uncensored policy (SMDP skills seed 8): `spearman` now returns NaN under 3 reached levels and `latex` writes no macro for an older file's 0.182; the write-up uses `spearman_score_fid256` instead (R106, [[Decision-tree interpretation]]).

## Tests

- `tests/interpretation/test_rules.py`: a planted 8-leaf policy over 5 decoded features plus 3 noise columns (recovered by 8 leaves; `rules_text` lists 8 leaves, uses only the 5 features, prints thresholds as integers and ordinal codes by name), leaves partitioning the states and playing what the tree predicts, merged intervals, `condition_text` for every kind of column, a code-less interval refused, the game read from the columns, `paths_to` by id and name with several chains for one action, importance ranking the 5 features above noise (memory columns in their own group), a known-truth importance of 0.75, seeds and column matching, the confusion by group against a hand count and fidelity per floor, `ACTION_GROUPS` covering the enum once, and the top level on a 40-key table (final checkpoint only, 4 skills, L90 = 4, the 10 highest-value rules in words, parquet input, `tried`, an empty checkpoint, one skill, no rows).
- `tests/interpretation/test_dataset.py`: seed splits, empty seeds refused, identical repeat collections with family and recurrent flag and outputs equal to `network_outputs` on an untrained Classic PPO checkpoint, steps since the last unlock, uniform subsampling, a recurrent toy whose labels follow its carry, a two-level toy's labels and skill columns, eval-file reuse, save and load, `relabel` of any rollout matching `collect` on the stochastic one.
- `tests/interpretation/test_distill.py`: a synthetic policy with a known 8-leaf rule over 5 features (top-1 1.0 from 8 leaves, L90 at most 8, expected fidelity equal to top-1 for a one-hot policy and by hand for a soft one), grid sizes and tree parameters, validation fidelity, worker counts giving identical trees (features, thresholds, node sizes), memory and skill column selection (a tree splitting on a skill column, curve flags), a label only a memory proxy explains, matching by column name, `complexity` on hand-made curves.
- `tests/interpretation/test_tree_policy.py`: actions on replayed states and 100 Classic steps of a 4-leaf tree through `play`, memory proxies online equal to `collect`'s offline ones with actions equal to scikit-learn's on those rows, both protocols playing the same episodes, errors for a missing reward (memory proxies, top level), uncomputable columns and an online unlock count off the events, a one-seed Craftax-Symbolic episode, a two-level toy keeping its skills, primitive actions and the reward and done it is handed, an `smdp-skills` checkpoint reading the played action and playing the tree's action on `collect`'s rows of learned skills, closed-loop fidelity 1 for a tree equal to its toy network, float32 splits at every threshold.
- `tests/interpretation/test_run_cli.py`: one CLI run (2 seeds per split, `--dagger`) over a temporary store and export with 2-update Classic `ppo` and `smdp-skills` runs and a `ppo` ablation: the method files of the plan and the skill files, tuned runs only at the final checkpoint, two fits per recurrent policy, only the 16- and 256-leaf trees kept, the median seed's rules, closed loop sharing one network run, the tried top level with its optimistic share, `complexity.csv` columns and rows, MLflow runs without datasets, resume and `--force`, censored medians and Spearman.
- `tests/interpretation/test_features.py`: round trips on edited reset states of both games, hand-placed targets and creatures, darkness, ladders, fountain and tree kinds, walkable blockers, Craftax creatures, width check, numpy vs jitted jax, 10,000 observations in whole chunks.
