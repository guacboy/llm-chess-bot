import chess
import random
import torch
from pathlib import Path

from agent.encoder import board_to_tensor
from agent.model import ChessNet, select_move

# __file__ is src/game.py - go up two levels to reach the project root.
DATA_PATH = Path(__file__).parent.parent / "data" / "games.pt"

# The bot always keeps at least this much randomness, even after many games.
MIN_EPSILON = 0.05

# How much the randomness drops after each game (5% per game).
EPSILON_DECAY = 0.05


def get_epsilon(games_played: int) -> float:
    """
    Returns the current exploration rate.
    Starts at 1.0 (fully random) and decays toward MIN_EPSILON over time.
    """
    return max(MIN_EPSILON, 1.0 - games_played * EPSILON_DECAY)


def get_bot_move(
    board: chess.Board,
    model: ChessNet,
    device: torch.device,
    epsilon: float,
) -> chess.Move:
    """
    The bot picks a move.
    With probability epsilon:      picks a random legal move.
    With probability 1 - epsilon:  uses the model to pick the best move.
    """
    if random.random() < epsilon:
        return random.choice(list(board.legal_moves))
    return select_move(model, board, board_to_tensor(board), device)


# ---------------------------------------------------------------------------
# Data persistence — saving and loading experience between sessions
# ---------------------------------------------------------------------------

def load_experiences() -> list[tuple[torch.Tensor, int, float]]:
    """Loads all saved game experience from disk. Returns empty list if none exists."""
    if DATA_PATH.exists():
        return torch.load(DATA_PATH, weights_only=False)
    return []


def save_experiences(
    new_experiences: list[tuple[torch.Tensor, int, float]],
) -> list[tuple[torch.Tensor, int, float]]:
    """
    Appends new experiences to the existing dataset and saves everything to disk.
    Returns the full combined dataset.
    """
    all_experiences = load_experiences() + new_experiences
    DATA_PATH.parent.mkdir(exist_ok=True)
    torch.save(all_experiences, DATA_PATH)
    return all_experiences
