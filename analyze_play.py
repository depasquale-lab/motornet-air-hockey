"""
Per-player behaviour summary for a trained checkpoint: is a player actually defending, or just idle?

  python analyze_play.py --game oct --ckpt octagon_hockey.pt
  python analyze_play.py --game 2p  --ckpt air_hockey.pt

Runs several recorded games (puck respawns after goals in the octagon game) and reports, per player,
over the frames where the puck is on that player's side:
  hand->puck    mean distance from the player's mallet to the puck
  idle->puck    same, if the mallet had stayed at its rest position (the "do nothing" baseline)
  track r       correlation between the mallet's and the puck's lateral position (own frame)
                (high = hand follows the puck sideways, i.e. blocking; ~0 = not responding)
and over all frames:
  speed         mean mallet speed (m/s)
  touches/min   number of separate mallet-puck contacts per minute
  goals vs/min  goals conceded per minute
A player that defends has hand->puck well below idle->puck and a clearly positive track r.
"""
import argparse

import numpy as np
import torch


def load(game, ckpt, device="cpu"):
    ck = torch.load(ckpt, map_location=device)
    if game == "oct":
        import motornet_octagon_hockey as G
        players = [G.Player(k, ck["hidden"], device) for k in range(G.N)]
        for k, pl in enumerate(players):
            pl.policy.load_state_dict(ck["policies"][k])
    else:
        import motornet_air_hockey as G
        players = [G.Player(0, ck["hidden"], device), G.Player(1, ck["hidden"], device)]
        players[0].policy.load_state_dict(ck["A"])
        players[1].policy.load_state_dict(ck["B"])
    return G, players, ck.get("it", "?")


def episode(G, game, players, T, hand_collide=False, hand_wall=False):
    with torch.no_grad():
        out = (G.rollout(players, 1, T, record=True, hand_collide=hand_collide, hand_wall=hand_wall) if game == "oct"
               else G.rollout(players[0], players[1], 1, T=T, record=True, hand_collide=hand_collide, hand_wall=hand_wall))
    H = out["hist"]
    puck = np.array(H["puck"])
    hands = np.array(H["hands"]) if game == "oct" else np.stack([np.array(H["A"]), np.array(H["B"])], 1)
    if game == "oct":
        return puck, hands, np.array(H["goals_against"])[-1]
    # 2-player game has no respawn: end the episode at the first goal (puck past an end line)
    goals = np.zeros(2)
    out_a, out_b = puck[:, 1] < G.Y_A - 0.02, puck[:, 1] > G.Y_B + 0.02
    gone = np.flatnonzero(out_a | out_b)
    if len(gone):
        goals[0 if out_a[gone[0]] else 1] = 1
        puck, hands = puck[: gone[0]], hands[: gone[0]]
    return puck, hands, goals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", choices=["oct", "2p"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--T", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hand_collide", action="store_true", help="prototype: mallets push off each other")
    ap.add_argument("--hand_wall", action="store_true", help="prototype: mallets are blocked by the table edge")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    G, players, it = load(args.game, args.ckpt)
    n = len(players)
    dt = G.DT
    contact = G.R_PUCK + G.R_MALLET
    rest = [G.pos_to_table(G.HAND_CENTER[None], k)[0].numpy() for k in range(n)]
    acc = {m: [[] for _ in range(n)] for m in ("hand_puck", "idle_puck", "hx", "px", "speed")}
    touches, goals, frames = np.zeros(n), np.zeros(n), 0

    for ep in range(args.episodes):
        puck, hands, g = episode(G, args.game, players, args.T, hand_collide=args.hand_collide, hand_wall=args.hand_wall)
        goals += g
        frames += len(puck)
        if args.game == "oct":
            own = (puck @ G.U.numpy().T).argmax(1)
        else:
            own = (puck[:, 1] > G.MID).astype(int)           # 0 = A's half
        for k in range(n):
            d = np.linalg.norm(hands[:, k] - puck, axis=1)
            touching = d < contact
            touches[k] += (touching[1:] & ~touching[:-1]).sum()
            if len(puck) < 3:
                continue
            acc["speed"][k].append(np.linalg.norm(np.diff(hands[:, k], axis=0), axis=1) / dt)
            m = own == k
            if m.sum() == 0:
                continue
            acc["hand_puck"][k].append(d[m])
            acc["idle_puck"][k].append(np.linalg.norm(rest[k] - puck[m], axis=1))
            hx = G.pos_from_table(torch.tensor(hands[m, k], dtype=torch.float32), k)[:, 0].numpy()
            px = G.pos_from_table(torch.tensor(puck[m], dtype=torch.float32), k)[:, 0].numpy()
            acc["hx"][k].append(hx)
            acc["px"][k].append(px)

    minutes = frames * dt / 60
    cat = lambda key, k: np.concatenate(acc[key][k]) if acc[key][k] else np.array([np.nan])
    rows = []
    for k in range(n):
        hx, px = cat("hx", k), cat("px", k)
        r = np.corrcoef(hx, px)[0, 1] if len(hx) > 2 and hx.std() > 0 and px.std() > 0 else np.nan
        rows.append((k, len(hx) / frames, cat("hand_puck", k).mean(), cat("idle_puck", k).mean(), r,
                     cat("speed", k).mean(), touches[k] / minutes, goals[k] / minutes))
    print(f"{args.ckpt} (iteration {it}), {args.episodes} games x {args.T * dt:.0f} s\n")
    print("player  puck-on-side  hand->puck  idle->puck  track r  speed  touches/min  goals vs/min")
    for k, frac, hp, ip, r, sp, tm, gm in rows:
        gtxt = f"{gm:>12.1f}"
        print(f"{k:>6}  {frac:>11.0%}  {hp:>10.3f}  {ip:>10.3f}  {r:>7.2f}  {sp:>5.2f}  {tm:>11.1f}  {gtxt}")
    print("\nhand->puck < idle->puck and track r > 0 means the player is moving to meet the puck;"
          "\nlow speed with track r ~ 0 means it is mostly idle.")


if __name__ == "__main__":
    main()
