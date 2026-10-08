# Run report: mallet-mallet and mallet-wall collisions, prototype (2026-10-08)

The README lists mallets not colliding with each other or the table edge as a known
simplification (only mallet-puck and puck-wall contacts were simulated). This adds both as
opt-in forces and compares against the baseline recipe on the 2-player game.

**Result: stable (no crashes or NaNs) but currently worse than baseline on every tracking
metric — the extra forces look like a bigger optimisation problem rather than a free win.**
Octagon side-by-side run still in progress, to be added to this report.

wandb: [2p-pretrain-tbptt-continued (baseline)](https://wandb.ai/bddepasq-boston-university/motornet-air-hockey/runs/5r5l1nba) · [2p-pretrain-tbptt-handcollide](https://wandb.ai/bddepasq-boston-university/motornet-air-hockey/runs/aakwxvfd)

| Baseline | Hand-collide |
|:---:|:---:|
| ![2p baseline](media/2p_baseline.gif) | ![2p hand-collide](media/2p_handcollide.gif) |

## What changed

`contact_force()` and `wall_force()` (both scripts) took a hardcoded puck radius; generalised
to accept a radius/radius-sum so the same spring-damper penalty force used for puck-mallet and
puck-wall contacts can be reused for mallet-mallet and mallet-wall. Two new opt-in flags, off by
default so existing behaviour and the already-running baseline jobs are untouched:

- `--hand_collide`: pairwise push force between mallets (`2*R_MALLET` apart triggers contact).
  In the octagon script this is vectorised over all `N*(N-1)` pairs (`hand_hand_force()`),
  with the diagonal masked out to avoid self-contact.
- `--hand_wall`: mallets are blocked by the same table-edge wall force the puck feels (still
  passes freely through the goal mouth, verified directly on synthetic positions).

Force directions were checked in isolation (overlapping mallets push apart symmetrically,
off-centre table-edge violations push back toward the centre, goal-mouth positions produce
~zero force) before training on them.

```
python motornet_air_hockey.py --iters 1000 --batch 256 --out_bias -2.5 \
  --pretrain_iters 600 --tbptt 30 --game_lr 5e-4 --curriculum_iters 400 \
  --chase_w0 10 --chase_floor 5 --effort 1e-3 --resume --hand_collide --hand_wall \
  --wandb motornet-air-hockey --run_name 2p-pretrain-tbptt-handcollide --video_every 150
```

Same recipe as the [2-player pretrain + tbptt run](2026-10-06_2p_pretrain_tbptt.md), with
`--hand_collide --hand_wall` added; baseline re-run alongside it for a same-day comparison.

## Results after 1000 iterations

|  | loss A | loss B | goals A | goals B | chase B | wallclock |
|---|---|---|---|---|---|---|
| baseline | 0.383 | 0.680 | 0.211 | 0.180 | 0.119 | 2h10m |
| hand-collide | 0.650 | 1.857 | 0.266 | 0.223 | 0.328 | 3h15m |

Reach-pretraining was unaffected either way (final dist ~0.02 m both arms, both runs).
Hand-collide is ~50% slower per iteration (the extra pairwise force) and worse on every game
loss term, most sharply on B's loss and the chase shaping weight.

## Behaviour analysis

`analyze_play.py`, 20 games of 4 s, with `--hand_collide --hand_wall` passed for the hand-collide
checkpoint so the evaluation physics match training.

| run | player | puck on side | hand->puck (m) | idle->puck (m) | track r | speed (m/s) | touches/min | goals against/min |
|---|---|---|---|---|---|---|---|---|
| baseline | 0 | 42% | 0.145 | 0.200 | 0.64 | 1.93 | 68.2 | 4.5 |
| baseline | 1 | 58% | 0.147 | 0.196 | 0.55 | 1.66 | 39.4 | 19.7 |
| hand-collide | 0 | 44% | 0.178 | 0.180 | 0.36 | 2.45 | 44.8 | 10.1 |
| hand-collide | 1 | 58% | 0.180 | 0.184 | 0.46 | 2.59 | 54.9 | 11.6 |

Baseline hands track the puck much more deliberately (r = 0.55-0.64, hand->puck well below
idle->puck). Hand-collide hands are faster but barely beat the idle baseline on distance and
track the puck more weakly (r = 0.36-0.46) — reads as arms spending effort reacting to each
other rather than to the puck, and player 1's goals-against roughly halved while still worse
than baseline, which doesn't look like a clean win in either direction.

## Caveats

- Single seed, one run per condition, no ablation of `--hand_collide` and `--hand_wall`
  separately — can't yet say which of the two forces is driving the regression.
- Octagon side-by-side (`octagon-pretrain-tbptt-handcollide` vs. the continued baseline) was
  still training when this report was written; training is running far slower than the ~8 s/iter
  benchmark on this cluster allocation (node contention), so it will be added separately.
- Contact stiffness/damping for the new forces reused the puck's `K_CONTACT`/`C_CONTACT`/`K_WALL`
  constants rather than being tuned independently.

## Next

- Ablate `--hand_collide` and `--hand_wall` independently to see which one hurts.
- If keeping either, retune contact stiffness/damping rather than reusing the puck's constants,
  and/or extend `--tbptt` truncation given the added stiff force terms.
- Add the octagon comparison once that pair of runs finishes.
