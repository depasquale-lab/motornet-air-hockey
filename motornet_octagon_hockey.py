"""
Eight MotorNet arms playing air hockey on an octagonal table.

Each player is a MotorNet RigidTendonArm26 (6 lumped Hill muscles, 2-DoF arm) driven by its
own GRU policy. One puck lives on a regular octagon; every side has a goal mouth and a
player standing behind it. Puck, wall and mallet contacts use smooth (softplus) penalty
forces, so the whole game -- muscles, skeletons, puck physics -- is differentiable and all
eight policies are trained end to end with BPTT, each descending its own loss. Contact
forces are fed back to each arm as an endpoint load, so the players feel the puck.

Geometry (table frame, metres, origin = table centre):
  Player k sits behind side k, at angle 45*k deg: shoulder at 0.60 * (cos, sin) of that angle,
  facing the centre. Side k is a wall at distance APOTHEM = 0.45 from the centre with a goal
  mouth of half-width 0.09 in its middle. Player k defends goal k.
  Each arm reaches ~0.64 m, i.e. just past the centre of the table.

Loss for player k (the sum over players is exactly zero):
  + goal conceded in own goal
  - (goals scored in the other 7 goals) / 7
  + territory: mean puck distance towards own side
  + chase shaping: distance from own mallet to puck, weighted by "puck is on my side"
  + small effort penalty

Usage:
  python motornet_octagon_hockey.py --resume                    # train
  python motornet_octagon_hockey.py --play octagon_hockey.pt    # render octagon_hockey.mp4

Notes / known simplifications
-----------------------------
- Mallets do not collide with each other and arms may overlap; only mallet-puck and
  puck-wall contacts are simulated.
- After the puck crosses a goal line the wall forces switch off for it (it has scored), so a
  scored puck cannot be flung into another goal. In --play mode the puck respawns at the
  centre after each goal and a "goals against" counter is shown beside each player.
- Same motornet quirk as the 2-player version: effector.to(device) does not update the
  muscle's/skeleton's own `device`, so Player.__init__ moves them explicitly.
- Each player's gradient also flows through the other players via the puck, and BPTT through
  repeated collisions is noisy. If training destabilises: shorter episodes, softer BETA,
  or detach the other players' observation inputs.
"""
import argparse
import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import motornet as mn

# ----------------------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------------------
N = 8                # players / sides
DT = 0.01            # effector timestep (s)
SUBSTEPS = 2         # puck physics substeps per effector step

APOTHEM = 0.45       # centre -> middle of each side
SHOULDER_R = 0.60    # centre -> shoulder
GOAL_HALF = 0.09     # goal mouth half-width

R_PUCK, R_MALLET = 0.025, 0.035
M_PUCK = 0.1
K_CONTACT, C_CONTACT = 600.0, 3.0   # mallet-puck spring / damper
K_WALL = 1500.0
PUCK_DRAG = 0.15
PUCK_NOISE = 0.2     # brownian kick on puck velocity (m/s per sqrt(s)); ~2 mm/s per 10 ms step
BETA = 300.0         # softplus sharpness (1/m) for contacts and walls
BETA_GOAL = 150.0    # sharpness of the soft goal indicator

PROP_DELAY = 2       # 20 ms proprioceptive delay
VIS_DELAY = 5        # 50 ms visual delay
LOAD_GAIN = 1.0      # scale on puck reaction force fed back to the arm

HAND_CENTER = torch.tensor([0.0, 0.45])   # hand rest position in the player's own frame

# ----------------------------------------------------------------------------------------
# Frames. Player k's own frame is the 2-link arm frame: shoulder at (0, 0), facing +y.
# In the table frame the player's +y axis points at the table centre.
# ----------------------------------------------------------------------------------------
_ANG = torch.arange(N, dtype=torch.float32) * (2 * np.pi / N)
U = torch.stack([_ANG.cos(), _ANG.sin()], -1)       # (N, 2) outward normal of side k
TAN = torch.stack([-_ANG.sin(), _ANG.cos()], -1)    # (N, 2) tangent along side k
EY = -U                                             # own +y in table coords
EX = torch.stack([EY[:, 1], -EY[:, 0]], -1)         # own +x (pure rotation of the A frame)
SHOULDER = SHOULDER_R * U


def pos_to_table(p, k):
    d = lambda c: c.to(p.device)
    return d(SHOULDER[k]) + p[..., 0:1] * d(EX[k]) + p[..., 1:2] * d(EY[k])


def vec_to_table(v, k):  # velocities and forces
    d = lambda c: c.to(v.device)
    return v[..., 0:1] * d(EX[k]) + v[..., 1:2] * d(EY[k])


def vec_from_table(v, k):
    return torch.stack([(v * EX[k].to(v.device)).sum(-1), (v * EY[k].to(v.device)).sum(-1)], -1)


def pos_from_table(p, k):
    return vec_from_table(p - SHOULDER[k].to(p.device), k)


# ----------------------------------------------------------------------------------------
# Differentiable puck physics
# ----------------------------------------------------------------------------------------
def soft_pen(x):
    """Smooth penetration depth: ~max(x, 0) with a ~3 mm rounded corner."""
    return F.softplus(BETA * x) / BETA


def contact_force(p, v, pm, vm):
    """Force on the puck from every mallet. p, v: (B, 1, 2); pm, vm: (B, N, 2).
    Returns (B, N, 2), the force each mallet exerts on the puck."""
    d = p - pm
    dist = torch.sqrt((d ** 2).sum(-1, keepdim=True) + 1e-8)
    n = d / dist
    overlap = (R_PUCK + R_MALLET) - dist
    pen = soft_pen(overlap)
    gate = torch.sigmoid(BETA * overlap)
    v_rel_n = ((v - vm) * n).sum(-1, keepdim=True)
    mag = K_CONTACT * pen - C_CONTACT * gate * v_rel_n
    return F.relu(mag) * n  # contacts only push


def wall_force(p):
    """Octagon walls with a goal gap in the middle of each side. p: (B, 2)."""
    proj = p @ U.T.to(p.device)                      # (B, N) distance along each outward normal
    tang = p @ TAN.T.to(p.device)                    # (B, N) position along each side
    outside_mouth = torch.sigmoid(BETA * (tang.abs() - GOAL_HALF))
    pen = soft_pen(proj - (APOTHEM - R_PUCK))        # how far the puck edge is into the wall
    # a puck that has crossed a goal line has scored: switch the walls off for it
    alive = torch.sigmoid(BETA_GOAL * (APOTHEM + 0.03 - proj.amax(-1, keepdim=True)))
    mag = K_WALL * outside_mouth * pen * alive       # (B, N)
    return -(mag.unsqueeze(-1) * U.to(p.device)).sum(1)


def puck_step(p, v, pm, vm):
    """Advance the puck one effector step. Mallet states (B, N, 2) are in the table frame.
    Returns new puck state and the mean force each mallet applied to the puck (B, N, 2)."""
    h = DT / SUBSTEPS
    f_acc = 0.0
    for s in range(SUBSTEPS):
        pm_s = pm + vm * h * s                        # mallets move linearly within the step
        fm = contact_force(p.unsqueeze(1), v.unsqueeze(1), pm_s, vm)
        f = fm.sum(1) + wall_force(p) - PUCK_DRAG * v
        v = v + h * f / M_PUCK + PUCK_NOISE * h ** 0.5 * torch.randn_like(v)  # semi-implicit Euler + jitter
        p = p + h * v
        f_acc = f_acc + fm
    return p, v, f_acc / SUBSTEPS


def goal_signal(P):
    """Soft indicator, per side, that the puck is past that end line inside its goal mouth.
    P: (..., 2) -> (..., N)"""
    proj = P @ U.T.to(P.device)
    tang = P @ TAN.T.to(P.device)
    return torch.sigmoid(BETA_GOAL * (GOAL_HALF - tang.abs())) * torch.sigmoid(BETA_GOAL * (proj - APOTHEM))


# ----------------------------------------------------------------------------------------
# Players
# ----------------------------------------------------------------------------------------
def make_arm():
    return mn.effector.RigidTendonArm26(muscle=mn.muscle.RigidTendonHillMuscle(), timestep=DT)


class Player:
    def __init__(self, idx, hidden=128, device="cpu"):
        self.idx = idx
        self.arm = make_arm().to(device)
        # motornet's .to() only updates the effector's own `device` attribute
        self.arm.muscle.to(device)
        self.arm.skeleton.to(device)
        # prop (12), own hand (2), puck pos/vel (4), the other N-1 hands (2 each), efference copy (6)
        self.n_obs = 12 + 2 + 4 + 2 * (N - 1) + 6
        self.policy = mn.policy.PolicyGRU(self.n_obs, hidden, self.arm.n_muscles, device=device)
        self.device = device

    def reset(self, batch):
        # shoulder ~45 deg, elbow ~90 deg, with jitter -> hand near (0, 0.45) in own frame
        q = np.deg2rad(np.array([45.0, 90.0])) + np.deg2rad(8.0) * np.random.randn(batch, 2)
        self.arm.reset(options={"batch_size": batch, "joint_state": torch.tensor(q, dtype=torch.float32)})
        self.h = self.policy.init_hidden(batch)
        self.u = torch.zeros(batch, self.arm.n_muscles, device=self.device)
        self.prop_buf = deque(maxlen=PROP_DELAY + 1)
        self.vis_buf = deque(maxlen=VIS_DELAY + 1)

    def hand_table(self):
        c = self.arm.states["cartesian"]
        return pos_to_table(c[:, :2], self.idx), vec_to_table(c[:, 2:], self.idx)

    def proprioception(self):
        m = self.arm.states["muscle"]
        mlen = m[:, 1:2, :] / self.arm.muscle.l0_ce   # same normalisation as mn.environment
        mvel = m[:, 2:3, :] / self.arm.muscle.vmax
        return torch.cat([mlen, mvel], dim=-1).squeeze(1)

    def observe(self, puck_p, puck_v, hands_table):
        """Egocentric, delayed observations. hands_table: (B, N, 2) mallet positions, table frame."""
        c = HAND_CENTER.to(puck_p.device)
        own = (self.arm.states["fingertip"] - c) / 0.3
        pp = (pos_from_table(puck_p, self.idx) - c) / 0.3
        pv = vec_from_table(puck_v, self.idx) / 2.0
        others = [(pos_from_table(hands_table[:, (self.idx + m) % N], self.idx) - c) / 0.3
                  for m in range(1, N)]   # ordered by position around the table, clockwise-relative
        prop, vis = self.proprioception(), torch.cat([own, pp, pv] + others, -1)
        if not self.prop_buf:  # fill buffers on first call
            self.prop_buf.extend([prop] * (PROP_DELAY + 1))
            self.vis_buf.extend([vis] * (VIS_DELAY + 1))
        self.prop_buf.append(prop)
        self.vis_buf.append(vis)
        return torch.cat([self.prop_buf[0], self.vis_buf[0], self.u], -1)

    def act(self, obs, noise=0.0):
        u, self.h = self.policy(obs, self.h)
        if noise > 0:
            u = (u + noise * torch.randn_like(u)).clamp(0, 1)
        self.u = u
        return u

    def step(self, u, load_on_hand_table):
        self.arm.step(u, endpoint_load=LOAD_GAIN * vec_from_table(load_on_hand_table, self.idx))


# ----------------------------------------------------------------------------------------
# Rollout
# ----------------------------------------------------------------------------------------
def init_puck(batch, device):
    """Puck near the centre with a random heading and speed."""
    r = 0.25 * torch.sqrt(torch.rand(batch, device=device))
    th = torch.rand(batch, device=device) * 2 * np.pi
    ang = torch.rand(batch, device=device) * 2 * np.pi
    spd = 0.2 + 1.3 * torch.rand(batch, device=device)
    p = torch.stack([r * torch.cos(th), r * torch.sin(th)], -1)
    v = torch.stack([spd * torch.cos(ang), spd * torch.sin(ang)], -1)
    return p, v


def rollout(players, batch, T, noise=0.0, record=False):
    """record=True: batch must be 1; the puck respawns at the centre after each goal and
    per-frame state is stored for rendering."""
    device = players[0].device
    for pl in players:
        pl.reset(batch)
    p, v = init_puck(batch, device)
    f = torch.zeros(batch, N, 2, device=device)
    hist = {"puck": [], "hands": [], "q": [], "goals_against": []}
    scores = np.zeros(N, dtype=int)
    traj_p, u_all, d_all = [], [], []

    for t in range(T):
        hp = torch.stack([pl.hand_table()[0] for pl in players], 1)
        us = [pl.act(pl.observe(p, v, hp), noise) for pl in players]
        # reaction force of the puck on each mallet (from the previous step's contacts)
        for k, pl in enumerate(players):
            pl.step(us[k], -f[:, k])
        hs = [pl.hand_table() for pl in players]
        hp = torch.stack([h[0] for h in hs], 1)
        hv = torch.stack([h[1] for h in hs], 1)
        p, v, f = puck_step(p, v, hp, hv)

        traj_p.append(p)
        u_all.append(torch.stack(us, 1))                       # (B, N, 6)
        d_all.append((hp - p.unsqueeze(1)).norm(dim=-1))       # (B, N)
        if record:
            g = goal_signal(p[0]).detach().cpu().numpy()
            hist["puck"].append(p[0].detach().cpu().numpy())
            hist["hands"].append(hp[0].detach().cpu().numpy())
            hist["q"].append(np.stack([pl.arm.states["joint"][0, :2].detach().cpu().numpy() for pl in players]))
            if g.max() > 0.5:   # goal: count it and respawn the puck
                scores[g.argmax()] += 1
                p, v = init_puck(1, device)
            hist["goals_against"].append(scores.copy())

    return dict(P=torch.stack(traj_p, 1), u=torch.stack(u_all, 1), d=torch.stack(d_all, 1), hist=hist)


def losses(out, w_terr=0.5, w_chase=1.0, w_effort=1e-2):
    """Returns per-player losses (N,) and a stats dict."""
    P = out["P"]                                               # (B, T, 2)
    proj = P @ U.T                                             # (B, T, N)
    goal = goal_signal(P).amax(1)                              # (B, N): goal conceded by player k
    scored_elsewhere = (goal.sum(-1, keepdim=True) - goal) / (N - 1)
    terr = -proj.clamp(-APOTHEM, APOTHEM).mean(1) / APOTHEM    # >0: puck far from my side
    zs = scored_elsewhere - goal + w_terr * terr               # zero-sum across players

    mine = torch.softmax(proj / 0.05, dim=-1)                  # soft "puck is on my side"
    chase = (mine * out["d"]).mean(1)                          # (B, N)
    effort = out["u"].pow(2).mean((1, 3))                      # (B, N)

    L = (-zs + w_chase * chase + w_effort * effort).mean(0)    # (N,)
    stats = dict(conceded=(goal > 0.5).float().mean().item() * N,   # goals per episode, all sides
                 chase=chase.mean().item())
    return L, stats


# ----------------------------------------------------------------------------------------
# Training: simultaneous gradient play, each net descends only its own loss
# ----------------------------------------------------------------------------------------
def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads) if args.threads else None
    players = [Player(k, args.hidden, args.device) for k in range(N)]
    params = [list(pl.policy.parameters()) for pl in players]
    opts = [torch.optim.Adam(ps, lr=args.lr) for ps in params]
    global SUBSTEPS
    SUBSTEPS = args.substeps
    start = 0
    if args.resume and os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location=args.device)
        for k in range(N):
            players[k].policy.load_state_dict(ck["policies"][k])
            opts[k].load_state_dict(ck["opts"][k])
        start = ck.get("it", 0)
        print(f"resumed from {args.ckpt} at iteration {start}", flush=True)
    elif args.resume:
        print(f"no checkpoint at {args.ckpt}, starting fresh", flush=True)

    for it in range(start, args.iters):
        # curriculum: lean on the chase shaping early, let the game take over later
        w_chase = max(0.2, 2.0 * (1 - it / (0.6 * args.iters)))
        out = rollout(players, args.batch, args.episode, noise=args.noise)
        L, st = losses(out, w_chase=w_chase)

        grads = [torch.autograd.grad(L[k], params[k], retain_graph=k < N - 1) for k in range(N)]
        for k in range(N):
            for prm, g in zip(params[k], grads[k]):
                prm.grad = g
            torch.nn.utils.clip_grad_norm_(params[k], 1.0)
            opts[k].step()

        if it % args.log_every == 0:
            print(f"it {it:5d} | goals/episode {st['conceded']:.2f} | chase {st['chase']:.3f} | "
                  f"loss " + " ".join(f"{x:+.2f}" for x in L.tolist()), flush=True)
        if (it + 1) % args.save_every == 0 or it == args.iters - 1:
            torch.save({"policies": [pl.policy.state_dict() for pl in players],
                        "opts": [o.state_dict() for o in opts],
                        "it": it + 1, "hidden": args.hidden}, args.ckpt + ".tmp")
            os.replace(args.ckpt + ".tmp", args.ckpt)   # never leave a half-written checkpoint
    return players


# ----------------------------------------------------------------------------------------
# Visualisation
# ----------------------------------------------------------------------------------------
def play(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    try:   # a bundled ffmpeg, if available
        import imageio_ffmpeg
        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

    global SUBSTEPS
    SUBSTEPS = args.substeps
    ck = torch.load(args.play, map_location=args.device)
    players = [Player(k, ck["hidden"], args.device) for k in range(N)]
    for k, pl in enumerate(players):
        pl.policy.load_state_dict(ck["policies"][k])
    with torch.no_grad():
        out = rollout(players, 1, args.T, record=True)
    H = {k: np.array(v) for k, v in out["hist"].items()}
    L1, L2 = players[0].arm.skeleton.L1, players[0].arm.skeleton.L2
    Un, EXn, EYn, SHn = U.numpy(), EX.numpy(), EY.numpy(), SHOULDER.numpy()
    colors = [plt.cm.tab10(i) for i in range(N)]

    def arm_pts(q, k):
        el = np.array([L1 * np.cos(q[0]), L1 * np.sin(q[0])])
        hd = el + np.array([L2 * np.cos(q.sum()), L2 * np.sin(q.sum())])
        pts = np.stack([np.zeros(2), el, hd])
        return SHn[k] + pts[:, 0:1] * EXn[k] + pts[:, 1:2] * EYn[k]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.axis("off")
    # octagon outline: vertices sit between neighbouring side midpoints
    vr = APOTHEM / np.cos(np.pi / N)
    va = (np.arange(N) + 0.5) * 2 * np.pi / N
    ax.add_patch(plt.Polygon(np.stack([vr * np.cos(va), vr * np.sin(va)], -1), fc="#e8f4fa", ec="k", lw=2))
    tex = []
    for k in range(N):
        m, t = APOTHEM * Un[k], np.stack([-Un[k][1], Un[k][0]])
        ax.plot(*np.stack([m - GOAL_HALF * t, m + GOAL_HALF * t]).T, color=colors[k], lw=6, solid_capstyle="butt")
        tex.append(ax.text(*(0.80 * Un[k]), "0", color=colors[k], ha="center", va="center", fontsize=14,
                           fontweight="bold"))
    arms = [ax.plot([], [], "-o", color=colors[k], lw=4, ms=4)[0] for k in range(N)]
    mallets = [ax.add_patch(plt.Circle((0, 0), R_MALLET, color=colors[k])) for k in range(N)]
    puck = ax.add_patch(plt.Circle((0, 0), R_PUCK, color="k", zorder=5))
    clock = ax.text(0, 0.97, "", ha="center", va="top", fontsize=10)
    ax.text(0, -0.97, "numbers = goals against each player", ha="center", va="bottom", fontsize=8, color="gray")

    def frame(t):
        for k in range(N):
            a = arm_pts(H["q"][t][k], k)
            arms[k].set_data(a[:, 0], a[:, 1])
            mallets[k].center = H["hands"][t][k]
            tex[k].set_text(str(H["goals_against"][t][k]))
        puck.center = H["puck"][t]
        clock.set_text(f"t = {t * DT:.2f} s")
        return arms + mallets + tex + [puck, clock]

    anim = animation.FuncAnimation(fig, frame, frames=len(H["puck"]), interval=1000 * DT, blit=True)
    try:
        anim.save(args.out + ".mp4", fps=int(1 / DT), dpi=100)
        print(f"saved {args.out}.mp4", flush=True)
    except Exception as e:
        print(f"mp4 failed ({e})", flush=True)
    anim.save(args.out + ".gif", fps=int(1 / DT) // 2, dpi=70)   # always keep a gif on disk too
    print(f"saved {args.out}.gif", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--episode", type=int, default=120, help="training episode length in 10 ms steps")
    ap.add_argument("--substeps", type=int, default=2, help="puck physics substeps per step")
    ap.add_argument("--resume", action="store_true", help="continue from --ckpt if it exists")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--noise", type=float, default=0.01, help="motor noise on muscle commands")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--ckpt", default="octagon_hockey.pt")
    ap.add_argument("--save_every", type=int, default=25)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--play", default=None, help="checkpoint to render instead of training")
    ap.add_argument("--T", type=int, default=600, help="steps to render in --play mode")
    ap.add_argument("--out", default="octagon_hockey", help="video basename in --play mode")
    args = ap.parse_args()
    play(args) if args.play else train(args)
