import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.model import ChessNet
from agent.trainer import load_model, clear_model
from agent.game import clear_experiences
from lichess.api import run_lichess_loop


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Chess Bot")
    parser.add_argument(
        "--reset-model",
        action="store_true",
        help="Delete saved model weights and exit. Bot will start from scratch next run.",
    )
    parser.add_argument(
        "--reset-data",
        action="store_true",
        help="Delete all saved game experience and exit. Dataset rebuilds from next game.",
    )
    parser.add_argument(
        "--reset-all",
        action="store_true",
        help="Delete both model weights and game experience, then exit.",
    )
    args = parser.parse_args()

    # --- Reset flags (exit after clearing, don't start the bot) ---
    if args.reset_all:
        clear_model()
        clear_experiences()
        return

    if args.reset_model:
        clear_model()
        return

    if args.reset_data:
        clear_experiences()
        return

    # --- Normal startup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ChessNet().to(device)
    load_model(model, device)

    run_lichess_loop(model, device)


if __name__ == "__main__":
    main()
