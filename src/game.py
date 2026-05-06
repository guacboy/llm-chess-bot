import chess
import random
import torch
from pathlib import Path

from encoder import board_to_tensor
from model import ChessNet, select_move, encode_move

# Where game experience data is saved between sessions.
DATA_PATH = Path("data/games.pt")

# A game that exceeds this many half-moves (plies) is called a draw.
# One ply = one move by one player. 200 plies = 100 full moves each.
MAX_PLIES = 200

# The bot always keeps at least this much randomness, even after many games.
MIN_EPSILON = 0.05

# How much the randomness drops after each game. (5% per game)
EPSILON_DECAY = 0.05


def get_epsilon(games_played: int) -> float:
    """
    Returns the current exploration rate.
    Starts at 1.0 (fully random) and decays toward MIN_EPSILON over time.
    """
    return max(MIN_EPSILON, 1.0 - games_played * EPSILON_DECAY)


def display_board(board: chess.Board, user_color: chess.Color) -> None:
    """
    Prints the board to the terminal with the user's pieces at the bottom.
    flipped=True rotates the board so Black's pieces appear at the bottom.
    """
    flipped = (user_color == chess.BLACK)
    print()
    print(board.unicode(borders=True, flipped=flipped))
    print()


def get_user_move(board: chess.Board) -> chess.Move | None:
    """
    Asks the user to type a move and keeps asking until a legal one is entered.
    Returns None if the user types 'quit'.
    """
    legal_ucis = {m.uci() for m in board.legal_moves}
    while True:
        raw = input("Your move (e.g. e2e4), or 'quit': ").strip().lower()
        if raw in ("quit", "q"):
            return None
        if raw in legal_ucis:
            return chess.Move.from_uci(raw)
        
        print(f"  Illegal move. Options: {', '.join(sorted(legal_ucis))}")


def get_bot_move(
    board: chess.Board,
    model: ChessNet,
    device: torch.device,
    epsilon: float,
) -> chess.Move:
    """
    The bot picks a move.
    With probability epsilon:       picks a random legal move.
    With probability 1 - epsilon:   uses the model to pick the best move.
    """
    if random.random() < epsilon:
        return random.choice(list(board.legal_moves))
    
    return select_move(model, board, board_to_tensor(board), device)


def get_outcome(board: chess.Board, user_color: chess.Color) -> float:
    """
    Converts the game result into a number from the user's perspective.
      +1.0 = user won
       0.0 = draw or unfinished
      -1.0 = user lost
    """
    result = board.result()  # "1-0", "0-1", "1/2-1/2", or "*" (unfinished)
    if result == "1-0":
        return 1.0 if user_color == chess.WHITE else -1.0
    if result == "0-1":
        return 1.0 if user_color == chess.BLACK else -1.0
    
    return 0.0


def play_game(
    model: ChessNet,
    device: torch.device,
    games_played: int,
) -> list[tuple[torch.Tensor, int, float]]:
    """
    Plays one full game between the user and the bot.

    Returns a list of experience tuples from the user's moves only:
      (board_tensor, move_index, outcome)
    These are what the trainer will learn from.
    """

    # --- Color selection ---
    # Default alternates each game so the dataset covers both sides evenly.
    default = "white" if games_played % 2 == 0 else "black"
    choice = input(f"\nPlay as White or Black? (press Enter for {default}): ").strip().lower()

    if choice in ("white", "w"):
        user_color = chess.WHITE
    elif choice in ("black", "b"):
        user_color = chess.BLACK
    else:
        user_color = chess.WHITE if default == "white" else chess.BLACK

    color_name = "White" if user_color == chess.WHITE else "Black"
    epsilon = get_epsilon(games_played)

    print(f"\nYou are playing as {color_name}.")
    print(f"Bot is {epsilon:.0%} random / {1 - epsilon:.0%} model-guided.")

    # --- Game loop ---
    board = chess.Board()

    # We collect (board_tensor, move_index) during the game.
    # The outcome gets attached after the game ends.
    user_records: list[tuple[torch.Tensor, int]] = []

    while not board.is_game_over():
        if board.ply() >= MAX_PLIES:
            print("\nMove limit reached - game is a draw.")
            break

        display_board(board, user_color)

        if board.turn == user_color:
            # --- User's turn ---
            move = get_user_move(board)
            if move is None:
                print("Game abandoned - no data saved.")
                return []
            
            # Record the board state and the move the user made, before pushing.
            user_records.append((board_to_tensor(board), encode_move(move)))
        else:
            # --- Bot's turn ---
            move = get_bot_move(board, model, device, epsilon)
            print(f"  Bot plays: {move.uci()}")

        board.push(move)

    # Show the final board position.
    display_board(board, user_color)

    # --- Outcome ---
    outcome = get_outcome(board, user_color)
    if outcome == 1.0:
        print("Result: You won!")
    elif outcome == -1.0:
        print("Result: You lost.")
    else:
        print("Result: Draw.")

    # Attach the outcome to every move the user made this game.
    return [(tensor, move_idx, outcome) for tensor, move_idx in user_records]


# ---------------------------------------------------------------------------
# Data persistence - saving and loading experience between sessions
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
