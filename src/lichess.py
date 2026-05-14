import os
import chess
import torch
import berserk
from dotenv import load_dotenv

from encoder import board_to_tensor
from model import ChessNet, encode_move
from game import get_epsilon, get_bot_move, load_experiences, save_experiences
from trainer import train, save_model

load_dotenv()

#TODO: have the bot play the best moves calculated from stockfish
# (to allow the user to play its best moves)

def get_client() -> berserk.Client:
    """Creates and returns an authenticated Lichess API client."""
    token = os.environ["LICHESS_BOT_API"]
    session = berserk.TokenSession(token)
    return berserk.Client(session)


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
    return 0.0  # draw or no winner field (stalemate, repetition, etc.)


def handle_game(
    client: berserk.Client,
    game_id: str,
    bot_username: str,
    model: ChessNet,
    device: torch.device,
    games_played: int,
) -> list[tuple[torch.Tensor, int, float]]:
    """
    Streams one Lichess game and handles it move by move.

    - When the user moves: records (board_tensor, move_index) for training.
    - When the bot moves: picks a move via get_bot_move and sends it to Lichess.
    - When the game ends: attaches the outcome to every recorded user move
      and returns the full list of experience tuples.
    """
    epsilon = get_epsilon(games_played)
    user_records: list[tuple[torch.Tensor, int]] = []
    bot_color: chess.Color | None = None
    user_color: chess.Color | None = None

    # last_move_count tracks how many moves we have already processed,
    # so we only act on genuinely new moves each time a gameState arrives.
    last_move_count = 0

    print(f"\n[Game {game_id}] Starting...")
    print(f"Bot: {epsilon:.0%} random / {1 - epsilon:.0%} model-guided\n")

    for event in client.bots.stream_game_state(game_id):

        # --- gameFull: sent once at the start with full game info ---
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

            # If it is already the bot's turn (bot plays White and goes first),
            # make the opening move immediately.
            if not board.is_game_over() and board.turn == bot_color:
                move = get_bot_move(board, model, device, epsilon)
                client.bots.make_move(game_id, move.uci())
                print(f"Bot plays: {move.uci()}")

        # --- gameState: sent every time a move is made or status changes ---
        elif event["type"] == "gameState":
            moves_str = event.get("moves", "")
            moves = moves_str.split() if moves_str else []
            status = event.get("status", "started")

            # Process only moves that arrived since the last event.
            new_moves = moves[last_move_count:]
            for i, move_uci in enumerate(new_moves):
                # Even-indexed moves (0, 2, 4…) are White's; odd are Black's.
                global_idx = last_move_count + i
                mover = chess.WHITE if global_idx % 2 == 0 else chess.BLACK

                if mover == user_color:
                    # The user just played this move - record it.
                    # We reconstruct the board BEFORE the move so we capture
                    # the position the user was looking at when they chose it.
                    board_before = reconstruct_board(" ".join(moves[:global_idx]))
                    user_move = chess.Move.from_uci(move_uci)
                    user_records.append(
                        (board_to_tensor(board_before), encode_move(user_move))
                    )
                    print(f"User played: {move_uci}")

            last_move_count = len(moves)
            board = reconstruct_board(moves_str)

            # Game over - attach outcome and return experiences.
            if status != "started":
                winner = event.get("winner")  # "white", "black", or absent (draw)
                outcome = get_outcome(winner, user_color)

                if outcome == 1.0:
                    print("Result: User won!")
                elif outcome == -1.0:
                    print("Result: User lost.")
                else:
                    print("Result: Draw.")

                return [(t, idx, outcome) for t, idx in user_records]

            # Still going - if it is now the bot's turn, send a move.
            if not board.is_game_over() and board.turn == bot_color:
                move = get_bot_move(board, model, device, epsilon)
                client.bots.make_move(game_id, move.uci())
                print(f"Bot plays: {move.uci()}")

    # Stream ended without a terminal status (rare - e.g. connection dropped).
    return [(t, idx, 0.0) for t, idx in user_records]


def run_lichess_loop(model: ChessNet, device: torch.device) -> None:
    """
    Main event loop. Connects to Lichess, accepts all incoming challenges,
    plays each game, then trains the model and saves everything after each game.
    """
    client = get_client()
    bot_username = client.account.get()["id"]
    print(f"Connected to Lichess as: {bot_username}")

    all_experiences = load_experiences()
    print(f"Loaded {len(all_experiences)} experiences from previous sessions.\n")
    print("Waiting for a challenge on Lichess...")
    print("Go to your bot's profile and challenge it from your regular account.\n")

    games_played = 0

    for event in client.bots.stream_incoming_events():

        if event["type"] == "challenge":
            challenge_id = event["challenge"]["id"]
            challenger = event["challenge"]["challenger"]["id"]
            print(f"Challenge from {challenger} - accepting...")
            client.bots.accept_challenge(challenge_id)

        elif event["type"] == "gameStart":
            game_id = event["game"]["id"]

            new_experiences = handle_game(
                client, game_id, bot_username, model, device, games_played
            )

            if new_experiences:
                games_played += 1
                all_experiences = save_experiences(new_experiences)
                print(f"\nSaved {len(new_experiences)} new moves. "
                      f"Total dataset: {len(all_experiences)} moves.")

                train(model, device, all_experiences)
                save_model(model)

            print("\nWaiting for next challenge...\n")
