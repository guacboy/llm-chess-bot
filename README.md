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
