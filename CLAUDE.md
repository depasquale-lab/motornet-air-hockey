# motornet-air-hockey — status note for Claude Code sessions

Read this first for current state and conventions, then `README.md` for the physics/game
design. This file is kept up to date after each major result so a new session (or the user
checking in) can get oriented fast without re-deriving everything from `git log`.

## Compute conventions (read before running anything)

- This runs on the BU SCC. **All training/rendering/evaluation goes through `qsub`, never run
  directly on the login node** — an interactive process over ~15 min CPU and >25% of its
  lifetime gets killed by the cluster's reaper. `jobs/*.qsub` holds one script per run as both
  the launcher and a reproducibility record of the exact command used.
- Chain dependent steps (e.g. "render + analyze + write report + push" after training finishes)
  with SGE's own `-hold_jid`, not by polling from an interactive session — the session can be
  closed or reaped without losing the chain. See `jobs/eval_2p_ablation.qsub` on
  `prototype-hand-collisions` for a worked example (held on two training jobs, then evals,
  appends a report section, commits, and pushes on its own).
- venv lives in `motornet-air-hockey/.venv` (python3.12, `pip install -r requirements.txt`).
  Other worktrees/branches of this repo can `source ../motornet-air-hockey/.venv/bin/activate`
  instead of reinstalling (~6 GB, mostly torch).
- SGE project: `-P rnn-models`. wandb project: `motornet-air-hockey`
  (https://wandb.ai/bddepasq-boston-university/motornet-air-hockey).
- Checkpoints (`*.pt`) and logs are gitignored as regenerable — only code, `jobs/*.qsub`,
  `reports/*.md`, and curated `media/*.gif` are committed.

## Cross-session progress tracking

Direct messaging between independent Claude Code sessions isn't available. The convention
used here (same one `odorant_receptor_search` uses):

- Every finished qsub chain commits and pushes a short summary as part of the job itself (not
  left to a human or a live session to write up afterward) — see the `git commit` step inside
  `jobs/eval_2p_ablation.qsub` for the pattern.
- `reports/<date>_<topic>.md` holds the narrative for each result (verdict up top, command
  used, metrics table, behaviour analysis via `analyze_play.py`, caveats, next steps). Read the
  most recent one first.
- This file (`CLAUDE.md`) gets updated after each major result/branch so "what's the current
  state" doesn't require reconstructing it from commits.
- For live Slack notification on pushes without any Claude session needing to stay open: a
  Slack workspace admin can run `/github subscribe depasquale-lab/motornet-air-hockey` after
  installing GitHub's own Slack app once — this is more robust than anything a Claude session
  can set up itself (a session-side workaround was tried first: a one-shot scheduled cloud
  routine via `RemoteTrigger`/the `schedule` skill, which works but needs re-creating per check
  and currently can't clone private context beyond what's connected).

## Current state (as of 2026-10-09)

**`main`**: baseline recipes from `reports/2026-10-06_2p_pretrain_tbptt.md` and
`reports/2026-10-06_octagon_pretrain_tbptt.md` were re-run start-to-finish as a fresh continuation
(no checkpoints existed to actually resume from — `*.pt` is gitignored). Both finished cleanly:
2p in ~2h10m, octagon in ~15.9h (much slower than the README's ~8s/iter benchmark on this
cluster allocation — node contention, not a code issue). wandb runs: `2p-pretrain-tbptt-continued`,
`octagon-pretrain-tbptt-continued`.

**`prototype-hand-collisions`** (not yet merged): adds opt-in `--hand_collide` /
`--hand_wall` flags — mallets were previously able to pass through each other and off the table
edge (a known simplification in the README). Generalizes `contact_force()`/`wall_force()` to a
parameterized radius so the puck's existing spring-damper contact model can be reused.

Findings so far (`reports/2026-10-08_hand_collision_prototype.md` on that branch, 2-player only):
- Combined (`--hand_collide --hand_wall`) is stable (no NaNs/crashes) but worse than baseline on
  every tracking metric (weaker puck-tracking correlation, ~50% slower per-iteration).
- Ablation isolating the two flags: **`--hand_wall` alone drives most of the regression**
  (loss B 0.73 vs. baseline 0.68 vs. 0.598 for `--hand_collide` alone vs. 1.857 combined) —
  mallet-wall blocking hurts more than mallet-mallet collision.
- Octagon side-by-side (`octagon-pretrain-tbptt-handcollide`, combined flags) also finished
  cleanly; goals-per-episode is higher than octagon baseline (0.836 vs. 0.516) but this hasn't
  been broken down with `analyze_play.py` yet — don't read that number alone as "better" or
  "worse" without the behavioural analysis.

**Open next steps**: retune contact stiffness/damping for the new forces instead of reusing the
puck's constants (prime suspect given `--hand_wall` alone is the bigger regression — it may be
too stiff/jarring on its own); run the octagon `analyze_play.py` comparison; decide whether to
merge the prototype or keep iterating on tuning first.
