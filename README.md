# LLM Chess Bot

A chess bot that learns your playstyle and plays it back against you on Lichess. The idea is that you are essentially **playing against yourself** - the bot observes your moves, learns your tendencies, and gradually mirrors them back at you to help you identify recurring patterns and mistakes.

---

## How it works

The bot starts by playing Stockfish's best move in every position. As you play more games, it shifts away from Stockfish and increasingly plays moves predicted by a neural network trained exclusively on **your** moves. By game 19, the bot plays your style 95% of the time.

**Learning signal:** After each game, every move you played is weighted by the outcome:
- Win → those moves get reinforced
- Loss → those moves get discouraged
- Draw → ignored

Over time the bot learns not just your style, but your *winning* style.

---

## Algorithm

### Model architecture — Feedforward Neural Network (MLP)

The model is a **Multi-Layer Perceptron** built in PyTorch with ~1.84 million trainable parameters.

**Input — board encoding (773 numbers)**

Every chess position is converted into a flat vector of 773 floating-point numbers:
- 12 binary 8×8 grids — one per piece type per colour (pawn, knight, bishop, rook, queen, king × white/black). Each cell is `1.0` if that piece occupies that square, `0.0` otherwise.
- 1 value for whose turn it is (`1.0` = white, `0.0` = black)
- 4 values for castling rights (one per right, `1.0` or `0.0`)

**Network layers**

```
Input       773  numbers  (board state)
    ↓  Linear + ReLU
Layer 1     512  neurons
    ↓  Linear + ReLU
Layer 2     512  neurons
    ↓  Linear + ReLU
Layer 3     256  neurons
    ↓  Linear
Output     4096  numbers  (one score per possible from→to move)
```

**Output — move selection**

The 4096 output scores cover every possible from-square/to-square combination (64 × 64). Before selecting a move, all illegal moves are masked to `-inf` and softmax is applied so the remaining scores sum to 1.0. The move with the highest probability is played.

---

### Learning algorithm — Outcome-Weighted Behavioural Cloning

The training approach combines two ideas:

**1. Behavioural cloning (supervised learning)**
The model is trained to predict *your* moves. For each position you faced, it is shown the board state and asked to assign high probability to the move you actually played. You are the teacher; your moves are the labels.

**2. Outcome weighting**
Pure behavioural cloning would copy your blunders just as readily as your good moves. Outcome weighting fixes this by scaling each training example by the game result:

| Outcome | Weight | Effect |
|---------|--------|--------|
| Win | +1.0 | Reinforce these moves — play them more |
| Draw | 0.0 | Ignore — no gradient update |
| Loss | −1.0 | Discourage these moves — play them less |

**Loss function**

For each move in a batch:
```
loss = −log_prob(move_played) × outcome
```

Averaged across the batch and minimised by the **Adam optimiser** (learning rate 0.001).

- When outcome is +1.0: the model is penalised for giving the move low probability → it learns to favour that move.
- When outcome is −1.0: the gradient is inverted → the model is pushed *away* from that move.
- When outcome is 0.0: no gradient, no change.

**Training schedule**

After every game, the full accumulated dataset is shuffled and trained for 5 epochs in batches of 64. Every new game causes the model to re-learn from all historical games, not just the most recent one.

---

## Requirements

- Python 3.10+
- NVIDIA GPU (recommended) or CPU
- A [Lichess](https://lichess.org) account for yourself
- A separate Lichess **bot** account (see setup below)
- [Stockfish](https://stockfishchess.org/download/) binary

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/guacboy/llm-chess-bot.git
cd llm-chess-bot
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install PyTorch (GPU)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

For CPU-only, omit the `--index-url` flag.

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5. Download Stockfish

1. Download the Windows binary from [stockfishchess.org/download](https://stockfishchess.org/download/)
2. Extract and place the `.exe` inside `src/stockfish/`

```
src/stockfish/stockfish-windows-x86-64-avx2.exe
```

The bot auto-detects any `.exe` in that folder at startup.

### 6. Create a Lichess bot account

1. Create a **new** Lichess account (do not use your main account - bot upgrade is irreversible)
2. In that account's settings, generate a **Personal API access token** with bot permissions
3. Create a `.env` file in the project root:

```
LICHESS_BOT_API=your_token_here
```

4. Upgrade the account to bot status (one-time, run from the project root):

```bash
python -c "
import os, requests
from dotenv import load_dotenv
load_dotenv()
r = requests.post('https://lichess.org/api/bot/account/upgrade',
    headers={'Authorization': f'Bearer {os.environ[\"LICHESS_BOT_API\"]}'})
print(r.json())
"
```

---

## Running the bot

```bash
venv\Scripts\python src\agent\main.py
```

Then go to your bot account's Lichess profile from your regular account and click **Challenge**. The terminal will display every move, the bot's move source (Stockfish vs learned style), and training progress after each game.

---

## Resetting

```bash
# Wipe learned weights only (if you want the bot to relearn from your existing games)
venv\Scripts\python src\agent\main.py --reset-model

# Wipe game data only (if your playstyle has changed significantly)
venv\Scripts\python src\agent\main.py --reset-data

# Full reset - start completely fresh
venv\Scripts\python src\agent\main.py --reset-all
```

---

## Project structure

```
llm-chess-bot/
├── src/
│   ├── agent/
│   │   ├── encoder.py   # Converts board position into a tensor (773 numbers)
│   │   ├── model.py     # Neural network (773 → 4096 move scores)
│   │   ├── trainer.py   # Trains model on game data, saves/loads weights
│   │   ├── game.py      # Epsilon decay, bot move selection, data persistence
│   │   └── main.py      # Entry point and CLI flags
│   ├── lichess/
│   │   └── api.py       # Lichess API loop, Stockfish integration, game streaming
│   └── stockfish/       # Place Stockfish .exe here (gitignored)
├── saved_models/        # Trained weights saved here (gitignored)
├── .env                 # Lichess API token (gitignored, never commit)
└── requirements.txt
```

---

## Bot behaviour over time

| Game | Stockfish | Learned style |
|------|-----------|---------------|
| 0    | 100%      | 0%            |
| 5    | 75%       | 25%           |
| 10   | 50%       | 50%           |
| 15   | 25%       | 75%           |
| 19+  | 5%        | 95%           |

To reach minimum Stockfish usage faster or slower, adjust `EPSILON_DECAY` in `src/agent/game.py`.
