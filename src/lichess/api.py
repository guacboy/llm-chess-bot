import os
import chess
import chess.engine
import torch
import berserk
from pathlib import Path
from dotenv import load_dotenv

from agent.encoder import board_to_tensor
from agent.model import ChessNet, encode_move
from agent.game import get_epsilon, get_bot_move, load_experiences, save_experiences
from agent.trainer import train, save_model

load_dotenv()


def get_client() -> berserk.Client:
    """Creates and returns an authenticated Lichess API client."""
    token = os.environ["LICHESS_BOT_API"]
    session = berserk.TokenSession(token)
    return berserk.Client(session)


def find_stockfish() -> str | None:
    """
    Looks for a Stockfish executable in src/stockfish/.
    Returns the path string if found, None otherwise.
    """
    # Auto-detect from the project's src/stockfish/ folder.
    sf_dir = Path(__file__).parent.parent / "stockfish"
    if sf_dir.exists():
        exes = list(sf_dir.glob("*.exe"))
        if exes:
            return str(exes[0])
    return None


def reconstruct_board(moves_str: str) -> chess.Board:
    """
    Rebuilds a chess.Board by replaying all moves from a space-separated UCI string.
    e.g. "e2e4 e7e5 g1f3" replays those three moves from the starting position.
    """
    board = chess.Board()
    if moves_str:
        for uci in moves_str.split():
            board.push(chess.Move.from_uci(uci))
    return board


def get_outcome(winner: str | None, user_color: chess.Color) -> float:
    """
    Converts the Lichess winner string into an outcome from the user's perspective.
      +1.0 = user won
       0.0 = draw or abandoned
      -1.0 = user lost
    """
    if winner == "white":
        return 1.0 if user_color == chess.WHITE else -1.0
    if winner == "black":
        return 1.0 if user_color == chess.BLACK else -1.0
    return 0.0


def handle_game(
    client: berserk.Client,
    game_id: str,
    bot_username: str,
    model: ChessNet,
    device: torch.device,
    games_played: int,
    sf_engine: chess.engine.SimpleEngine | None,
) -> list[tuple[torch.Tensor, int, float]]:
    """
    Streams one Lichess game and handles it move by move.

    - When the user moves: records (board_tensor, move_index) for training.
    - When the bot moves: uses get_bot_move (Stockfish or model) and sends it.
    - When the game ends: attaches the outcome and prints a move-source breakdown.
    """
    epsilon = get_epsilon(games_played)
    user_records: list[tuple[torch.Tensor, int]] = []
    bot_color: chess.Color | None = None
    user_color: chess.Color | None = None
    last_move_count = 0

    # Track how many bot moves came from each source this game.
    move_counts = {"model": 0, "stockfish": 0, "random": 0}

    print(f"\n[Game {game_id}] Starting...")
    print(f"Bot: {1 - epsilon:.0%} model / {epsilon:.0%} Stockfish\n")

    for event in client.bots.stream_game_state(game_id):

        # --- gameFull: sent once at the start ---
        if event["type"] == "gameFull":
            white_id = event["white"].get("id", "")
            bot_color = chess.WHITE if white_id == bot_username else chess.BLACK
            user_color = chess.BLACK if bot_color == chess.WHITE else chess.WHITE

            color_name = "White" if bot_color == chess.WHITE else "Black"
            print(f"Bot is playing as {color_name}")

            moves_str = event["state"].get("moves", "")
            moves = moves_str.split() if moves_str else []
            last_move_count = len(moves)
            board = reconstruct_board(moves_str)

            if not board.is_game_over() and board.turn == bot_color:
                move, source = get_bot_move(board, model, device, epsilon, sf_engine)
                move_counts[source] += 1
                client.bots.make_move(game_id, move.uci())
                print(f"Bot plays: {move.uci()}  [{source}]")

        # --- gameState: sent on every move or status change ---
        elif event["type"] == "gameState":
            moves_str = event.get("moves", "")
            moves = moves_str.split() if moves_str else []
            status = event.get("status", "started")

            new_moves = moves[last_move_count:]
            for i, move_uci in enumerate(new_moves):
                global_idx = last_move_count + i
                mover = chess.WHITE if global_idx % 2 == 0 else chess.BLACK

                if mover == user_color:
                    board_before = reconstruct_board(" ".join(moves[:global_idx]))
                    user_move = chess.Move.from_uci(move_uci)
                    user_records.append(
                        (board_to_tensor(board_before), encode_move(user_move))
                    )
                    print(f"User played: {move_uci}")

            last_move_count = len(moves)
            board = reconstruct_board(moves_str)

            # Game over - print summary and return experiences.
            if status != "started":
                winner = event.get("winner")
                outcome = get_outcome(winner, user_color)

                if outcome == 1.0:
                    print("\nResult: User won!")
                elif outcome == -1.0:
                    print("\nResult: User lost.")
                else:
                    print("\nResult: Draw.")

                # --- Move source breakdown ---
                total_bot_moves = sum(move_counts.values())
                if total_bot_moves > 0:
                    model_pct  = move_counts["model"]      / total_bot_moves * 100
                    sf_pct     = move_counts["stockfish"]  / total_bot_moves * 100
                    print(f"\nBot move breakdown ({total_bot_moves} moves):")
                    print(f"  Learned user style : {move_counts['model']:3d}  ({model_pct:.0f}%)")
                    print(f"  Stockfish          : {move_counts['stockfish']:3d}  ({sf_pct:.0f}%)")
                    if model_pct >= sf_pct:
                        print("  >> Bot is playing more like you than Stockfish now.")
                    else:
                        print("  >> Bot still leans on Stockfish - keep playing to train it.")

                return [(t, idx, outcome) for t, idx in user_records]

            # Still going - send bot's next move.
            if not board.is_game_over() and board.turn == bot_color:
                move, source = get_bot_move(board, model, device, epsilon, sf_engine)
                move_counts[source] += 1
                client.bots.make_move(game_id, move.uci())
                print(f"Bot plays: {move.uci()}  [{source}]")

    return [(t, idx, 0.0) for t, idx in user_records]


def run_lichess_loop(model: ChessNet, device: torch.device) -> None:
    """
    Main event loop. Initialises Stockfish, then connects to Lichess and
    accepts challenges indefinitely. After each game, trains the model and saves.
    """
    # --- Stockfish setup ---
    sf_path = find_stockfish()
    sf_engine: chess.engine.SimpleEngine | None = None
    try:
        sf_engine = chess.engine.SimpleEngine.popen_uci(sf_path)
        print(f"Stockfish loaded: {sf_path}")
    except Exception as e:
        print(f"Stockfish not available ({e}) - falling back to random moves.")

    # --- Lichess setup ---
    client = get_client()
    bot_username = client.account.get()["id"]
    print(f"Connected to Lichess as: {bot_username}")

    all_experiences = load_experiences()
    print(f"Loaded {len(all_experiences)} experiences from previous sessions.\n")
    print("Waiting for a challenge on Lichess...")
    print("Go to your bot's profile and challenge it from your regular account.\n")

    games_played = 0

    try:
        for event in client.bots.stream_incoming_events():

            if event["type"] == "challenge":
                challenge_id = event["challenge"]["id"]
                challenger = event["challenge"]["challenger"]["id"]
                print(f"Challenge from {challenger} - accepting...")
                client.bots.accept_challenge(challenge_id)

            elif event["type"] == "gameStart":
                game_id = event["game"]["id"]

                new_experiences = handle_game(
                    client, game_id, bot_username,
                    model, device, games_played, sf_engine,
                )

                if new_experiences:
                    games_played += 1
                    all_experiences = save_experiences(new_experiences)
                    print(f"\nSaved {len(new_experiences)} new moves. "
                          f"Total dataset: {len(all_experiences)} moves.")
                    train(model, device, all_experiences)
                    save_model(model)

                print("\nWaiting for next challenge...\n")

    finally:
        # Always close the Stockfish process cleanly when the program exits.
        if sf_engine is not None:
            sf_engine.quit()
