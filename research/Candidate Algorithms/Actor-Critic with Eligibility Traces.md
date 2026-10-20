---
title: "Actor-Critic with Eligibility Traces"
type: candidate
status: candidate
parent: "[[Candidate Algorithms]]"
tags:
  - candidate
  - craftax
  - algorithm-family
  - actor-critic
  - eligibility-traces
  - online-learning
origin: "Sutton & Barto, Reinforcement Learning: An Introduction, Ch. 11 (actor-critic with eligibility traces); Elsayed, Vasan, Mahmood. Streaming Deep Reinforcement Learning Finally Works. 2024. https://arxiv.org/abs/2410.14606"
created: 2026-10-20
updated: 2026-10-20
---
# Actor-Critic with Eligibility Traces

## Idea in plain words

- The textbook online actor-critic: after every single game step, compute the critic's TD error $\delta$ and update both the actor (the policy) and the critic (the value function) at once.
- TD error: $\delta = r + \gamma V(s') - V(s)$, "how much better the step went than the critic expected".
- Eligibility trace: each network keeps a decaying sum of its recent gradients. One $\delta$ then updates the weights behind the last roughly $1/(1-\gamma\lambda)$ steps, not only the last one.
- No rollout buffer, no minibatches, no epochs, no clipping, no GAE: the opposite of PPO's batch update.
- With many parallel envs, each env keeps its own traces, and the update at each step is the average over envs.
- The same machinery gives Watkins's Q($\lambda$) with traces, the value-based twin. Together with PPO and PQN this fills a 2 x 2: batch or online, policy gradient or value-based.

## Why it fits Craftax

- Long episodes: up to 100,000 steps ([[Craftax episode and time]]). PPO's GAE stops at the 64-step rollout edge and bootstraps there ([[Craftax Baselines algorithms and hyperparameters]]); a trace never stops, it fades smoothly.
- Rewards far apart: an achievement reward flows back through the trace to the actions that set it up (collecting wood long before placing a table), without storing those steps.
- Survival reward is dense: 0.1 x health change every step ([[Craftax scoring and evaluation]], [[Craftax survival stats]]), so the critic gets signal every step to learn from online.
- The world keeps changing as the agent goes deeper (new floors, new creatures, [[Craftax floors and progression]]); an online learner adapts at every step instead of every update.
- Parallel JAX envs: averaging $\delta \cdot z$ over hundreds of envs removes much of the noise that made single-stream deep actor-critic unstable.

## How it works

Builds on the one-step actor-critic and actor-critic with eligibility traces ([[11 Policy approximation (actor-critic)]]), the backward view of TD($\lambda$) ([[07a n-step TD and TD(λ)]]) and Watkins's Q($\lambda$) for the twin ([[07b Eligibility-trace control and off-policy]]).

For each environment $i$ at each step:

1. Act: $a \sim \pi_\theta(\cdot \mid s)$; observe $r$, $s'$, done.
2. TD error: $\delta = r + \gamma (1 - \text{done})\, V_w(s') - V_w(s)$.
3. Critic trace: $z_w \leftarrow \gamma\lambda_w z_w + \nabla_w V_w(s)$.
4. Actor trace: $z_\theta \leftarrow \gamma\lambda_\theta z_\theta + \nabla_\theta \log \pi_\theta(a \mid s)$.
5. Update with the mean over envs: $w \leftarrow w + \alpha_w\,\overline{\delta z_w}$, $\theta \leftarrow \theta + \alpha_\theta\,\overline{\delta z_\theta} + \alpha_\theta\,\beta_H \nabla_\theta H(\pi)$.
6. On episode end, reset that env's traces to zero.

```
delta   = r + gamma * (1 - done) * V(s2) - V(s)              # shape (envs,)
z_w     = gamma * lam * z_w + grad_w(V(s))                    # per env
z_th    = gamma * lam * z_th + grad_th(log_pi(a | s))         # per env
w      += alpha_w  * mean_envs(delta * z_w)
theta  += alpha_th * mean_envs(delta * z_th) + alpha_th * beta_H * grad_entropy
z_w, z_th = where(done, 0, z_w), where(done, 0, z_th)
```

- Stabilisers from streaming deep RL (Elsayed et al.): LayerNorm in each layer, sparse initialisation, running observation and reward scaling, and a bound on the step size relative to $|\delta|$ and the trace size. Each is a few lines.
- Q($\lambda$) twin: a Q-network instead of $V$, ε-greedy actions, target $r + \gamma \max_{a'} Q(s', a')$, and the trace cut to zero after a non-greedy action (Watkins).

## How to implement here

- Base: `src/wut_murl_craftax/training/ppo.py` for environment setup, the `ActorCritic` network and run records; replace the rollout-then-learn loop (`_make_rollout`, `_make_learn`) with one `jax.lax.scan` over steps whose carry holds parameters, traces, optimizer state and env state.
- Per-env gradients: `jax.vmap(jax.grad(f))` over the env axis; traces are pytrees with a leading env axis; the update is `tree_map(lambda z: (delta[:, None, ...] * z).mean(0), z)`.
- Trace arrays have shape `(num_envs, *param_shape)`; shard envs over devices with `jax.pmap` or `shard_map` and average updates with `pmean`.
- Start feed-forward (comparable with [[PPO (Matthews 2024)]]); recurrent version: a GRU with one-step truncated gradients (the previous hidden state enters as a constant, `stop_gradient`).
- New code: about 200-300 lines.
- Extra hyperparameters: $\alpha_\theta$, $\alpha_w$, $\lambda_\theta$, $\lambda_w$ (0.8-0.95), $\gamma$ (0.99 and 0.999), entropy weight $\beta_H$, number of envs, step-size bound.

## Risks

- Online deep actor-critic is known to be unstable; Elsayed et al. report that classic streaming methods "often fail to learn" without their fixes, so the stabilisers are needed from day one.
- Truncated gradients in the GRU version may learn memory poorly compared with PPO-RNN's backpropagation through the rollout.
- Samples within one env are strongly correlated; averaging across envs is the only decorrelation.
- Step sizes are sensitive: there is no clipping to stop one bad step.
- Entropy may collapse early without PPO's many-sample normalised advantages; tune $\beta_H$.

## Novelty check

- Searched (2026-10-20): Semantic Scholar citations of arXiv 2402.16801 (106 papers, filtered for actor-critic, eligibility, trace, streaming, online); arXiv full text of Elsayed et al. (2410.14606) searched for "Craftax" and "Crafter" (no hits; experiments on 58 Atari games and 50 continuous-control tasks from MuJoCo and DM Control); web "streaming deep RL actor-critic eligibility traces Craftax OR Crafter"; GitHub repository search "craftax actor-critic", "craftax a2c".
- On Craftax every policy-gradient agent uses batch updates (PPO variants in [[Existing Implementations]], REINFORCE with a baseline only inside Simulus's imagination).
- Not found: an online actor-critic or Q($\lambda$) with eligibility traces on Craftax or Crafter.
- New here: the book's backward-view actor-critic on Craftax, with per-env traces over parallel JAX envs, and its Q($\lambda$) twin as the value-based partner.

## How to evaluate

- Feed-forward version: Craftax-1B against [[PPO (Matthews 2024)]] (11.9%, block A) and Craftax-1M against block D (2.2%).
- Recurrent version: against PPO-RNN (15.3% at 1B, 2.3% at 1M); Q($\lambda$) twin against [[PQN-RNN (Gallici 2025)]] (16.0% at 1B).
- Score: native reward as % of 226, our own reruns as controls, same seeds, 5+ seeds.
- Achievement success, with a focus on achievements far from their prerequisites (`MAKE_IRON_PICKAXE`, `ENTER_DUNGEON`), and mean survival time.
- The 2 x 2 table: PPO (batch PG), PQN (batch value), this note (online PG), its twin (online value), at matched steps.

## References

- Sutton, R. S., Barto, A. G. Reinforcement Learning: An Introduction, Chapter 11 (draft numbering; Chapter 13 in the 2018 edition).
- Elsayed, M., Vasan, G., Mahmood, A. R. "Streaming Deep Reinforcement Learning Finally Works." 2024. https://arxiv.org/abs/2410.14606
- Gallici, M. et al. "Simplifying Deep Temporal Difference Learning." ICLR 2025. https://arxiv.org/abs/2407.04811
- Matthews, M. et al. "Craftax." ICML 2024. https://arxiv.org/abs/2402.16801
