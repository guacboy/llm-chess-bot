import sys
import torch

# Add src/ to the path so all modules can find each other.
sys.path.insert(0, "src")

from model import ChessNet
from game import play_game, load_experiences, save_experiences
from trainer import train, save_model, load_model


def main() -> None:
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ChessNet().to(device)
    load_model(model, device)

    # Load the full experience dataset accumulated so far.
    all_experiences = load_experiences()
    print(f"Loaded {len(all_experiences)} experiences from previous sessions.\n")

    games_played = 0

    # --- Main loop ---
    while True:
        print(f"{'=' * 40}")
        print(f"  Game {games_played + 1}")
        print(f"{'=' * 40}")

        # 1. Play a game and collect the user's moves as training data.
        new_experiences = play_game(model, device, games_played)

        if new_experiences:
            games_played += 1

            # 2. Save the new experiences to disk and merge with the full dataset.
            all_experiences = save_experiences(new_experiences)
            print(f"Saved {len(new_experiences)} new moves. "
                  f"Total dataset: {len(all_experiences)} moves.")

            # 3. Train the model on everything collected so far.
            train(model, device, all_experiences)

            # 4. Save the updated model weights.
            save_model(model)

        # 5. Ask whether to play again.
        again = input("\nPlay another game? (y/n): ").strip().lower()
        if again not in ("y", "yes"):
            print("\nGoodbye! Your progress has been saved.")
            break


if __name__ == "__main__":
    main()
