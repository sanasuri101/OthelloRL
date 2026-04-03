# Remotion Demo Video Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 45-second clean/technical Remotion video for Twitter/LinkedIn showing the Othello PPO+LSTM RL agent's architecture, training results, self-play, and win rates.

**Architecture:** Python script captures real move sequences from checkpoint_016600 as JSON → Remotion React/TypeScript compositions animate them as SVG board with piece-by-piece animation → six scenes compose into a single 45s MP4.

**Tech Stack:** Remotion 4.x, React, TypeScript, Python (move capture), SVG animations

---

## File Structure

```
video/
├── package.json                   # Remotion + dependencies
├── tsconfig.json                  # TypeScript config
├── remotion.config.ts             # Remotion config (fps=30, dimensions)
├── src/
│   ├── Root.tsx                   # Registers OthelloDemo composition
│   ├── OthelloDemo.tsx            # Main 45s composition, sequences all scenes
│   ├── scenes/
│   │   ├── Hook.tsx               # Scene 1 (0-3s): hook text fade-in
│   │   ├── Architecture.tsx       # Scene 2 (3-8s): pipeline diagram
│   │   ├── CurriculumChart.tsx    # Scene 3 (8-15s): win rate bar chart
│   │   ├── SelfPlayGame.tsx       # Scene 4 (15-30s): live board animation
│   │   ├── WinRates.tsx           # Scene 5 (30-38s): final stats
│   │   └── CTA.tsx                # Scene 6 (38-45s): GitHub CTA
│   ├── components/
│   │   ├── OthelloBoard.tsx       # SVG 8x8 board + pieces
│   │   ├── Piece.tsx              # Single piece with scale-in animation
│   │   └── BarChart.tsx           # Animated horizontal bar chart
│   └── data/
│       └── moves.json             # Real move sequence from checkpoint_016600
scripts/
└── capture_moves.py               # Runs agent headlessly, outputs moves.json
```

---

## Chunk 1: Project Setup + Move Capture

### Task 1: Scaffold Remotion project

**Files:**
- Create: `video/package.json`
- Create: `video/tsconfig.json`
- Create: `video/remotion.config.ts`

- [ ] **Step 1: Create video directory and package.json**

```bash
mkdir -p /Users/openclaw/Code/Connect4RL/video/src/scenes
mkdir -p /Users/openclaw/Code/Connect4RL/video/src/components
mkdir -p /Users/openclaw/Code/Connect4RL/video/src/data
cd /Users/openclaw/Code/Connect4RL/video
```

```json
{
  "name": "othello-rl-demo",
  "version": "1.0.0",
  "scripts": {
    "start": "npx remotion studio",
    "render": "npx remotion render OthelloDemo out/demo.mp4",
    "build": "npx remotion render OthelloDemo out/demo.mp4 --codec=h264"
  },
  "dependencies": {
    "@remotion/cli": "^4.0.0",
    "remotion": "^4.0.0",
    "react": "^18.0.0",
    "react-dom": "^18.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.0.0",
    "typescript": "^5.0.0"
  }
}
```

- [ ] **Step 2: Create tsconfig.json**

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["dom", "ES2020"],
    "jsx": "react-jsx",
    "module": "commonjs",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true
  },
  "include": ["src"]
}
```

- [ ] **Step 3: Create remotion.config.ts**

```ts
import { Config } from "@remotion/cli/config";

Config.setVideoImageFormat("jpeg");
Config.setOverwriteOutput(true);
```

- [ ] **Step 4: Install dependencies**

```bash
cd /Users/openclaw/Code/Connect4RL/video && npm install
```

Expected: `node_modules/` created, no errors.

- [ ] **Step 5: Commit**

```bash
git add video/
git commit -m "chore: scaffold remotion video project"
```

---

### Task 2: Capture real move sequences from checkpoint_016600

**Files:**
- Create: `scripts/capture_moves.py`
- Create: `video/src/data/moves.json`

- [ ] **Step 1: Create capture_moves.py**

```python
"""
Run checkpoint_016600 in self-play (both sides), capture move sequences as JSON.
Output: video/src/data/moves.json
"""
import sys, json, torch, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "othello"))

import othello as _othello_pkg
import gymnasium as gym

OBS_DIM = 192
NUM_ACTIONS = 65

class _FakeEnv:
    single_observation_space = gym.spaces.Box(
        low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
    )
    single_action_space = gym.spaces.Discrete(NUM_ACTIONS)

def load_policy(ckpt_path):
    from othello.train import make_policy
    policy = make_policy(_FakeEnv(), hidden_size=256)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy

def fresh_lstm(h):
    return {
        "lstm_h": torch.zeros(1, 1, h),
        "lstm_c": torch.zeros(1, 1, h),
    }

def board_state(env):
    """Return 64-element list: 0=empty, 1=black, 2=white."""
    obs = env.observations[0]
    black = obs[0:64]
    white = obs[64:128]
    state = []
    for i in range(64):
        if black[i] > 0.5:
            state.append(1)
        elif white[i] > 0.5:
            state.append(2)
        else:
            state.append(0)
    return state

def pick_action(policy, obs_t, lstm_state):
    with torch.no_grad():
        logits, _ = policy.forward_eval(obs_t, lstm_state)
        legal = obs_t[:, 128:192]
        has_legal = legal.sum(dim=-1, keepdim=True) > 0
        pass_legal = (~has_legal).float()
        full_mask = torch.cat([legal, pass_legal], dim=-1)
        logits = logits.masked_fill(full_mask < 0.5, float("-inf"))
        return int(logits.argmax(dim=-1).item())

def capture_game(policy, env):
    env.reset()
    h = policy.lstm_cell.hidden_size
    black_lstm = fresh_lstm(h)
    white_lstm = fresh_lstm(h)
    moves = []

    while True:
        # Black's turn
        obs_t = torch.tensor(env.observations, dtype=torch.float32)
        action = pick_action(policy, obs_t, black_lstm)
        moves.append({
            "color": "black",
            "square": action,
            "board_before": board_state(env),
        })
        opp_obs = env.step_agent(np.array([action], dtype=np.int32))

        opp_obs_t = torch.tensor(opp_obs, dtype=torch.float32)
        white_action = pick_action(policy, opp_obs_t, white_lstm)
        moves.append({
            "color": "white",
            "square": white_action,
            "board_before": board_state(env),
        })
        _, rew, term, _, _ = env.step_opponent(
            np.array([white_action], dtype=np.int32)
        )

        if term[0]:
            black_score = int(sum(1 for x in board_state(env) if x == 1))
            white_score = int(sum(1 for x in board_state(env) if x == 2))
            result = "black" if rew[0] > 0 else ("white" if rew[0] < 0 else "draw")
            return {
                "moves": moves,
                "result": result,
                "final_score": {"black": black_score, "white": white_score},
                "board_final": board_state(env),
            }

def main():
    ckpt = "experiments/othello_ppo/best/checkpoint_016600.pt"
    policy = load_policy(ckpt)
    env = _othello_pkg.Othello(num_envs=1)

    print("Capturing 3 games...")
    games = []
    for i in range(3):
        game = capture_game(policy, env)
        print(f"  Game {i+1}: {game['result']} "
              f"({game['final_score']['black']}-{game['final_score']['white']})")
        games.append(game)

    out = Path("video/src/data/moves.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"games": games}, indent=2))
    print(f"Saved to {out}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script from project root**

```bash
cd /Users/openclaw/Code/Connect4RL
uv run python scripts/capture_moves.py
```

Expected output:
```
Capturing 3 games...
  Game 1: black (50-14)
  Game 2: ...
Saved to video/src/data/moves.json
```

- [ ] **Step 3: Verify moves.json has correct structure**

```bash
python -c "
import json
d = json.load(open('video/src/data/moves.json'))
g = d['games'][0]
print('Moves:', len(g['moves']))
print('Result:', g['result'])
print('Score:', g['final_score'])
print('First move square:', g['moves'][0]['square'])
"
```

Expected: moves count ~50-60, result black/white/draw, score two numbers summing to ≤64.

- [ ] **Step 4: Commit**

```bash
git add scripts/capture_moves.py video/src/data/moves.json
git commit -m "feat: add move capture script and self-play game data"
```

---

## Chunk 2: Core Components

### Task 3: OthelloBoard + Piece components

**Files:**
- Create: `video/src/components/Piece.tsx`
- Create: `video/src/components/OthelloBoard.tsx`

- [ ] **Step 1: Create Piece.tsx**

```tsx
// video/src/components/Piece.tsx
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import React from "react";

type Props = {
  color: "black" | "white";
  appearAtFrame: number;
};

export const Piece: React.FC<Props> = ({ color, appearAtFrame }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const scale = spring({
    fps,
    frame: frame - appearAtFrame,
    config: { damping: 12, stiffness: 200, mass: 0.5 },
    durationInFrames: 12,
  });

  const fill = color === "black" ? "#1a1a2e" : "#f0f0f0";
  const stroke = color === "black" ? "#4a4a8a" : "#cccccc";

  return (
    <circle
      r={22}
      fill={fill}
      stroke={stroke}
      strokeWidth={2}
      style={{ transform: `scale(${scale})`, transformOrigin: "center" }}
    />
  );
};
```

- [ ] **Step 2: Create OthelloBoard.tsx**

```tsx
// video/src/components/OthelloBoard.tsx
import React from "react";
import { useCurrentFrame } from "remotion";
import { Piece } from "./Piece";

type Move = {
  color: "black" | "white";
  square: number;
  board_before: number[];
};

type Props = {
  moves: Move[];
  framesPerMove: number;
  size: number;
};

export const OthelloBoard: React.FC<Props> = ({ moves, framesPerMove, size }) => {
  const frame = useCurrentFrame();
  const cellSize = size / 8;
  const pieceRadius = cellSize * 0.42;

  // Determine which moves have been played so far
  const movesPlayed = Math.min(
    Math.floor(frame / framesPerMove),
    moves.length
  );

  // Build board state from moves
  const board = Array(64).fill(0);
  // Initial 4 pieces
  board[27] = 2; board[28] = 1; board[35] = 1; board[36] = 2;

  const piecesWithFrames: Array<{ sq: number; color: "black" | "white"; frame: number }> = [
    { sq: 27, color: "white", frame: -999 },
    { sq: 28, color: "black", frame: -999 },
    { sq: 35, color: "black", frame: -999 },
    { sq: 36, color: "white", frame: -999 },
  ];

  moves.slice(0, movesPlayed).forEach((move, i) => {
    if (move.square < 64) {
      const appearFrame = i * framesPerMove;
      // Simulate board after this move (simplified: just track placed pieces)
      piecesWithFrames.push({
        sq: move.square,
        color: move.color,
        frame: appearFrame,
      });
    }
  });

  return (
    <svg width={size} height={size}>
      {/* Board background */}
      <rect width={size} height={size} fill="#155724" rx={4} />

      {/* Grid lines */}
      {Array.from({ length: 9 }).map((_, i) => (
        <React.Fragment key={i}>
          <line
            x1={i * cellSize} y1={0}
            x2={i * cellSize} y2={size}
            stroke="#0d3d18" strokeWidth={1}
          />
          <line
            x1={0} y1={i * cellSize}
            x2={size} y2={i * cellSize}
            stroke="#0d3d18" strokeWidth={1}
          />
        </React.Fragment>
      ))}

      {/* Pieces */}
      {piecesWithFrames.map(({ sq, color, frame: appearFrame }, i) => {
        const row = Math.floor(sq / 8);
        const col = sq % 8;
        const cx = col * cellSize + cellSize / 2;
        const cy = row * cellSize + cellSize / 2;
        return (
          <g key={i} transform={`translate(${cx}, ${cy})`}>
            <Piece color={color} appearAtFrame={appearFrame} />
          </g>
        );
      })}
    </svg>
  );
};
```

- [ ] **Step 3: Commit**

```bash
git add video/src/components/
git commit -m "feat: add OthelloBoard and Piece SVG components"
```

---

### Task 4: BarChart component

**Files:**
- Create: `video/src/components/BarChart.tsx`

- [ ] **Step 1: Create BarChart.tsx**

```tsx
// video/src/components/BarChart.tsx
import React from "react";
import { interpolate, useCurrentFrame } from "remotion";

type Bar = {
  label: string;
  value: number; // 0-1
  color: string;
  startFrame: number;
};

type Props = {
  bars: Bar[];
  width: number;
  height: number;
};

export const BarChart: React.FC<Props> = ({ bars, width, height }) => {
  const frame = useCurrentFrame();
  const barHeight = 44;
  const gap = 20;
  const labelWidth = 120;
  const maxBarWidth = width - labelWidth - 80;

  return (
    <svg width={width} height={height}>
      {bars.map((bar, i) => {
        const y = i * (barHeight + gap);
        const progress = interpolate(
          frame - bar.startFrame,
          [0, 20],
          [0, 1],
          { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
        );
        const barWidth = bar.value * progress * maxBarWidth;
        const pct = Math.round(bar.value * 100 * progress);

        return (
          <g key={i} transform={`translate(0, ${y})`}>
            {/* Label */}
            <text
              x={labelWidth - 10}
              y={barHeight / 2 + 5}
              textAnchor="end"
              fill="#e0e0e0"
              fontSize={18}
              fontFamily="monospace"
            >
              {bar.label}
            </text>
            {/* Bar background */}
            <rect
              x={labelWidth}
              y={0}
              width={maxBarWidth}
              height={barHeight}
              fill="#1e1e2e"
              rx={4}
            />
            {/* Bar fill */}
            <rect
              x={labelWidth}
              y={0}
              width={barWidth}
              height={barHeight}
              fill={bar.color}
              rx={4}
            />
            {/* Percentage */}
            <text
              x={labelWidth + barWidth + 8}
              y={barHeight / 2 + 5}
              fill="#ffffff"
              fontSize={18}
              fontFamily="monospace"
              fontWeight="bold"
            >
              {pct}%
            </text>
          </g>
        );
      })}
    </svg>
  );
};
```

- [ ] **Step 2: Commit**

```bash
git add video/src/components/BarChart.tsx
git commit -m "feat: add animated BarChart component"
```

---

## Chunk 3: Scenes

### Task 5: Hook scene (0-3s)

**Files:**
- Create: `video/src/scenes/Hook.tsx`

- [ ] **Step 1: Create Hook.tsx**

```tsx
// video/src/scenes/Hook.tsx
import React from "react";
import { interpolate, useCurrentFrame } from "remotion";

export const Hook: React.FC = () => {
  const frame = useCurrentFrame();

  const opacity = interpolate(frame, [0, 15], [0, 1], {
    extrapolateRight: "clamp",
  });
  const y = interpolate(frame, [0, 15], [20, 0], {
    extrapolateRight: "clamp",
  });

  return (
    <div style={{
      width: "100%", height: "100%",
      background: "#0d0d1a",
      display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      fontFamily: "monospace",
    }}>
      <div style={{ opacity, transform: `translateY(${y}px)`, textAlign: "center" }}>
        <div style={{ fontSize: 22, color: "#888", marginBottom: 16, letterSpacing: 4 }}>
          REINFORCEMENT LEARNING
        </div>
        <div style={{ fontSize: 52, color: "#ffffff", fontWeight: "bold", lineHeight: 1.2 }}>
          I trained an Othello agent
        </div>
        <div style={{ fontSize: 52, color: "#4ade80", fontWeight: "bold" }}>
          from scratch
        </div>
      </div>
    </div>
  );
};
```

- [ ] **Step 2: Commit**

```bash
git add video/src/scenes/Hook.tsx
git commit -m "feat: add Hook scene"
```

---

### Task 6: Architecture scene (3-8s)

**Files:**
- Create: `video/src/scenes/Architecture.tsx`

- [ ] **Step 1: Create Architecture.tsx**

```tsx
// video/src/scenes/Architecture.tsx
import React from "react";
import { interpolate, useCurrentFrame } from "remotion";

const steps = [
  { label: "C Bitboard Engine", sub: "game logic + negamax", color: "#60a5fa" },
  { label: "PPO + LSTM", sub: "policy gradient", color: "#a78bfa" },
  { label: "Curriculum Training", sub: "random → negamax d5 → self-play", color: "#34d399" },
  { label: "150M steps", sub: "~3hrs on A100", color: "#fbbf24" },
];

export const Architecture: React.FC = () => {
  const frame = useCurrentFrame();

  return (
    <div style={{
      width: "100%", height: "100%",
      background: "#0d0d1a",
      display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      fontFamily: "monospace",
      gap: 0,
    }}>
      <div style={{ fontSize: 28, color: "#888", marginBottom: 48, letterSpacing: 3 }}>
        ARCHITECTURE
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 0 }}>
        {steps.map((step, i) => {
          const appear = interpolate(frame - i * 8, [0, 12], [0, 1], {
            extrapolateLeft: "clamp", extrapolateRight: "clamp",
          });
          return (
            <React.Fragment key={i}>
              <div style={{
                opacity: appear,
                transform: `scale(${0.8 + 0.2 * appear})`,
                background: "#1e1e2e",
                border: `2px solid ${step.color}`,
                borderRadius: 8,
                padding: "16px 24px",
                textAlign: "center",
                minWidth: 180,
              }}>
                <div style={{ color: step.color, fontSize: 20, fontWeight: "bold" }}>
                  {step.label}
                </div>
                <div style={{ color: "#888", fontSize: 14, marginTop: 6 }}>
                  {step.sub}
                </div>
              </div>
              {i < steps.length - 1 && (
                <div style={{
                  opacity: appear,
                  color: "#444",
                  fontSize: 32,
                  margin: "0 8px",
                }}>→</div>
              )}
            </React.Fragment>
          );
        })}
      </div>
    </div>
  );
};
```

- [ ] **Step 2: Commit**

```bash
git add video/src/scenes/Architecture.tsx
git commit -m "feat: add Architecture scene"
```

---

### Task 7: CurriculumChart scene (8-15s)

**Files:**
- Create: `video/src/scenes/CurriculumChart.tsx`

- [ ] **Step 1: Create CurriculumChart.tsx**

```tsx
// video/src/scenes/CurriculumChart.tsx
import React from "react";
import { useCurrentFrame } from "remotion";
import { BarChart } from "../components/BarChart";

const bars = [
  { label: "Random", value: 1.00, color: "#4ade80", startFrame: 0 },
  { label: "Negamax d1", value: 1.00, color: "#4ade80", startFrame: 12 },
  { label: "Negamax d2", value: 0.87, color: "#60a5fa", startFrame: 24 },
  { label: "Negamax d3", value: 1.00, color: "#4ade80", startFrame: 36 },
  { label: "Negamax d5", value: 0.84, color: "#fbbf24", startFrame: 48 },
];

export const CurriculumChart: React.FC = () => {
  return (
    <div style={{
      width: "100%", height: "100%",
      background: "#0d0d1a",
      display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      fontFamily: "monospace",
    }}>
      <div style={{ fontSize: 28, color: "#888", marginBottom: 48, letterSpacing: 3 }}>
        WIN RATE vs NEGAMAX
      </div>
      <BarChart bars={bars} width={860} height={320} />
    </div>
  );
};
```

- [ ] **Step 2: Commit**

```bash
git add video/src/scenes/CurriculumChart.tsx
git commit -m "feat: add CurriculumChart scene"
```

---

### Task 8: SelfPlayGame scene (15-30s)

**Files:**
- Create: `video/src/scenes/SelfPlayGame.tsx`

- [ ] **Step 1: Create SelfPlayGame.tsx**

```tsx
// video/src/scenes/SelfPlayGame.tsx
import React from "react";
import { useCurrentFrame } from "remotion";
import { OthelloBoard } from "../components/OthelloBoard";
import movesData from "../data/moves.json";

export const SelfPlayGame: React.FC = () => {
  const frame = useCurrentFrame();
  const game = movesData.games[0];
  const framesPerMove = 8; // ~3.75 moves/sec at 30fps

  const movesPlayed = Math.min(
    Math.floor(frame / framesPerMove),
    game.moves.length
  );

  const score = { black: 2, white: 2 };
  game.moves.slice(0, movesPlayed).forEach((m) => {
    if (m.square < 64) {
      if (m.color === "black") score.black++;
      else score.white++;
    }
  });

  return (
    <div style={{
      width: "100%", height: "100%",
      background: "#0d0d1a",
      display: "flex", alignItems: "center", justifyContent: "center",
      fontFamily: "monospace",
      gap: 64,
    }}>
      <div style={{ textAlign: "center" }}>
        <div style={{ fontSize: 22, color: "#888", marginBottom: 24, letterSpacing: 3 }}>
          SELF-PLAY — checkpoint_016600
        </div>
        <OthelloBoard
          moves={game.moves as any}
          framesPerMove={framesPerMove}
          size={420}
        />
        <div style={{ marginTop: 24, display: "flex", gap: 48, justifyContent: "center" }}>
          <div style={{ color: "#ccc", fontSize: 20 }}>
            ⚫ Black: <span style={{ color: "#fff", fontWeight: "bold" }}>{score.black}</span>
          </div>
          <div style={{ color: "#ccc", fontSize: 20 }}>
            ⚪ White: <span style={{ color: "#fff", fontWeight: "bold" }}>{score.white}</span>
          </div>
        </div>
      </div>
    </div>
  );
};
```

- [ ] **Step 2: Commit**

```bash
git add video/src/scenes/SelfPlayGame.tsx
git commit -m "feat: add SelfPlayGame scene with live board animation"
```

---

### Task 9: WinRates + CTA scenes (30-45s)

**Files:**
- Create: `video/src/scenes/WinRates.tsx`
- Create: `video/src/scenes/CTA.tsx`

- [ ] **Step 1: Create WinRates.tsx**

```tsx
// video/src/scenes/WinRates.tsx
import React from "react";
import { interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";

const stats = [
  { label: "vs Negamax d1", value: "100%", color: "#4ade80" },
  { label: "vs Negamax d2", value: "87%", color: "#60a5fa" },
  { label: "vs Negamax d3", value: "100%", color: "#4ade80" },
  { label: "vs Negamax d5", value: "84%", color: "#fbbf24" },
];

export const WinRates: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <div style={{
      width: "100%", height: "100%",
      background: "#0d0d1a",
      display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      fontFamily: "monospace",
      gap: 32,
    }}>
      <div style={{ fontSize: 28, color: "#888", letterSpacing: 3, marginBottom: 16 }}>
        EVALUATION RESULTS
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 32 }}>
        {stats.map((stat, i) => {
          const scale = spring({
            fps, frame: frame - i * 10,
            config: { damping: 14, stiffness: 180 },
            durationInFrames: 20,
          });
          return (
            <div key={i} style={{
              transform: `scale(${scale})`,
              background: "#1e1e2e",
              border: `2px solid ${stat.color}`,
              borderRadius: 12,
              padding: "28px 40px",
              textAlign: "center",
            }}>
              <div style={{ color: stat.color, fontSize: 52, fontWeight: "bold" }}>
                {stat.value}
              </div>
              <div style={{ color: "#888", fontSize: 18, marginTop: 8 }}>
                {stat.label}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
```

- [ ] **Step 2: Create CTA.tsx**

```tsx
// video/src/scenes/CTA.tsx
import React from "react";
import { interpolate, useCurrentFrame } from "remotion";

export const CTA: React.FC = () => {
  const frame = useCurrentFrame();
  const opacity = interpolate(frame, [0, 20], [0, 1], {
    extrapolateRight: "clamp",
  });

  return (
    <div style={{
      width: "100%", height: "100%",
      background: "#0d0d1a",
      display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      fontFamily: "monospace",
      opacity,
    }}>
      <div style={{ fontSize: 42, color: "#ffffff", fontWeight: "bold", marginBottom: 24 }}>
        Open Source
      </div>
      <div style={{ fontSize: 22, color: "#4ade80", marginBottom: 48 }}>
        github.com/sanasuri101/OthelloRL
      </div>
      <div style={{ fontSize: 18, color: "#888" }}>
        C bitboard engine · PPO+LSTM · Curriculum RL · Raylib rendering
      </div>
    </div>
  );
};
```

- [ ] **Step 3: Commit**

```bash
git add video/src/scenes/WinRates.tsx video/src/scenes/CTA.tsx
git commit -m "feat: add WinRates and CTA scenes"
```

---

## Chunk 4: Composition + Render

### Task 10: Wire all scenes into main composition

**Files:**
- Create: `video/src/OthelloDemo.tsx`
- Create: `video/src/Root.tsx`

- [ ] **Step 1: Create OthelloDemo.tsx**

```tsx
// video/src/OthelloDemo.tsx
import React from "react";
import { AbsoluteFill, Series } from "remotion";
import { Hook } from "./scenes/Hook";
import { Architecture } from "./scenes/Architecture";
import { CurriculumChart } from "./scenes/CurriculumChart";
import { SelfPlayGame } from "./scenes/SelfPlayGame";
import { WinRates } from "./scenes/WinRates";
import { CTA } from "./scenes/CTA";

// 30fps. Durations in frames.
const FPS = 30;
const scenes = [
  { component: Hook,            durationS: 3  },  // 0-3s
  { component: Architecture,    durationS: 5  },  // 3-8s
  { component: CurriculumChart, durationS: 7  },  // 8-15s
  { component: SelfPlayGame,    durationS: 15 },  // 15-30s
  { component: WinRates,        durationS: 8  },  // 30-38s
  { component: CTA,             durationS: 7  },  // 38-45s
];

export const OthelloDemo: React.FC = () => {
  return (
    <AbsoluteFill style={{ background: "#0d0d1a" }}>
      <Series>
        {scenes.map(({ component: Scene, durationS }, i) => (
          <Series.Sequence key={i} durationInFrames={durationS * FPS}>
            <Scene />
          </Series.Sequence>
        ))}
      </Series>
    </AbsoluteFill>
  );
};
```

- [ ] **Step 2: Create Root.tsx**

```tsx
// video/src/Root.tsx
import React from "react";
import { Composition } from "remotion";
import { OthelloDemo } from "./OthelloDemo";

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="OthelloDemo"
      component={OthelloDemo}
      durationInFrames={45 * 30}  // 45 seconds at 30fps
      fps={30}
      width={1280}
      height={720}
    />
  );
};
```

- [ ] **Step 3: Commit**

```bash
git add video/src/OthelloDemo.tsx video/src/Root.tsx
git commit -m "feat: wire all scenes into 45s OthelloDemo composition"
```

---

### Task 11: Preview and render

**Files:** none new

- [ ] **Step 1: Open Remotion Studio to preview**

```bash
cd /Users/openclaw/Code/Connect4RL/video
npm run start
```

Open `http://localhost:3000` — scrub through all 6 scenes, check timing and animations.

- [ ] **Step 2: Fix any timing issues**

Adjust `durationS` values in `OthelloDemo.tsx` if scenes feel rushed or slow.

- [ ] **Step 3: Render to MP4**

```bash
mkdir -p out
npm run render
```

Expected: `video/out/demo.mp4` ~45 seconds, 1280x720, H.264.

- [ ] **Step 4: Check output**

```bash
ffprobe video/out/demo.mp4 2>&1 | grep -E "Duration|Video"
```

Expected: `Duration: 00:00:45`, `Video: h264, 1280x720`.

- [ ] **Step 5: Final commit**

```bash
git add video/out/demo.mp4
git commit -m "feat: render final 45s Othello RL demo video"
```

---

## Quick Reference — Run Commands

```bash
# Capture move data from agent
cd /Users/openclaw/Code/Connect4RL
uv run python scripts/capture_moves.py

# Install video dependencies
cd video && npm install

# Preview in browser
npm run start

# Render to MP4
npm run render
```
