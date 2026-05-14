import sys
import torch

sys.path.insert(0, "src")

from model import ChessNet
from trainer import load_model
from lichess import run_lichess_loop


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ChessNet().to(device)
    load_model(model, device)

    run_lichess_loop(model, device)


if __name__ == "__main__":
    main()
