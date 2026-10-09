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


## Ablation: isolating `--hand_collide` vs `--hand_wall` (2026-10-08 follow-up)

Same 2p recipe, run with each flag alone to see which force drives the regression reported above.

| condition | loss A | loss B | goals A | goals B | chase B |
|---|---|---|---|---|---|
| baseline | 0.383 | 0.68 | 0.211 | 0.18 | 0.119 |
| --hand_collide only | 0.432 | 0.598 | 0.25 | 0.19 | 0.102 |
| --hand_wall only | 0.94 | 0.73 | 0.27 | 0.26 | 0.151 |
| both (combined) | 0.65 | 1.857 | 0.266 | 0.223 | 0.328 |

Isolating the two forces: loss B is 0.598 with `--hand_collide` alone and 0.73 with `--hand_wall` alone, against 0.68 (baseline) and 1.857 (combined). **mallet-wall blocking** accounts for more of the regression.

| `--hand_collide` only | `--hand_wall` only |
|:---:|:---:|
| ![2p hand-collide only](media/2p_handcollide_only.gif) | ![2p hand-wall only](media/2p_handwall_only.gif) |

### Behaviour analysis, `--hand_collide` only

```
air_hockey_handcollide_only.pt (iteration 1000), 20 games x 4 s

player  puck-on-side  hand->puck  idle->puck  track r  speed  touches/min  goals vs/min
     0          72%       0.144       0.191     0.60   1.54         58.7          12.0
     1          28%       0.149       0.210     0.54   1.81         29.4           5.3

hand->puck < idle->puck and track r > 0 means the player is moving to meet the puck;
low speed with track r ~ 0 means it is mostly idle.
```

### Behaviour analysis, `--hand_wall` only

```
air_hockey_handwall_only.pt (iteration 1000), 20 games x 4 s

player  puck-on-side  hand->puck  idle->puck  track r  speed  touches/min  goals vs/min
     0          62%       0.169       0.196     0.46   2.65         69.7           7.1
     1          38%       0.167       0.205     0.49   2.94         63.8           8.3

hand->puck < idle->puck and track r > 0 means the player is moving to meet the puck;
low speed with track r ~ 0 means it is mostly idle.
```


## Octagon: baseline vs. hand-collide (2026-10-09 follow-up)

`analyze_play.py` for the octagon pair referenced in the "Current state" note — baseline
(`octagon-pretrain-tbptt-continued`) vs. combined `--hand_collide --hand_wall`
(`octagon-pretrain-tbptt-handcollide`), both 1000 iterations. `goals/episode` alone looked worse
for hand-collide (0.836 vs. 0.516); the per-player breakdown below is what that number is
actually made of.

| Baseline | Hand-collide |
|:---:|:---:|
| ![octagon baseline](media/octagon_baseline.gif) | ![octagon hand-collide](media/octagon_handcollide.gif) |

### Octagon baseline

```
octagon_hockey.pt (iteration 1000), 20 games x 4 s

player  puck-on-side  hand->puck  idle->puck  track r  speed  touches/min  goals vs/min
     0          13%       0.203       0.139    -0.01   2.59         31.5           3.8
     1          12%       0.176       0.158     0.14   2.62         17.2           3.8
     2          10%       0.204       0.157     0.06   1.55         10.5           4.5
     3           9%       0.205       0.153     0.27   3.26         30.8           8.2
     4          11%       0.229       0.169     0.18   1.98         16.5           8.2
     5          16%       0.181       0.165     0.28   2.84         28.5           6.0
     6          14%       0.203       0.168     0.15   2.77         39.0           4.5
     7          15%       0.254       0.156     0.07   2.48         20.2           5.2

hand->puck < idle->puck and track r > 0 means the player is moving to meet the puck;
low speed with track r ~ 0 means it is mostly idle.
```

### Octagon hand-collide (`--hand_collide --hand_wall`)

```
octagon_hockey_handcollide.pt (iteration 1000), 20 games x 4 s

player  puck-on-side  hand->puck  idle->puck  track r  speed  touches/min  goals vs/min
     0           8%       0.189       0.166     0.21   2.47         27.8           7.5
     1          11%       0.194       0.177     0.30   2.49         26.2           6.0
     2          12%       0.281       0.162     0.07   3.28         36.0           6.0
     3          12%       0.241       0.153     0.15   3.31         24.8           6.8
     4          16%       0.474       0.476    -0.02   2.95         36.0           9.0
     5          16%       0.310       0.156     0.05   2.47         27.0           9.0
     6          13%       0.207       0.144     0.12   2.82         11.2           7.5
     7          12%       0.205       0.143    -0.02   3.37         26.2           6.8

hand->puck < idle->puck and track r > 0 means the player is moving to meet the puck;
low speed with track r ~ 0 means it is mostly idle.
```
