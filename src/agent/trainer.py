import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from agent.model import ChessNet

# __file__ is src/agent/trainer.py - go up three levels to reach the project root.
MODEL_PATH = Path(__file__).parent.parent.parent / "saved_models" / "model.pt"

#TODO: add an option to clear the model's data

def train(
    model: ChessNet,
    device: torch.device,
    experiences: list[tuple[torch.Tensor, int, float]],
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.001,
) -> None:
    """
    Trains the model on all saved experiences.

    For each (board_tensor, move_index, outcome) triple:
      - The model predicts a probability for every possible move.
      - We look at how likely it rated the move the user actually played.
      - We multiply that score by the outcome:
          win  (+1.0) → reinforce this move
          draw ( 0.0) → ignore (no gradient)
          loss (-1.0) → push the model away from this move
      - We do this across many examples in batches, then update the weights.
    """

    if len(experiences) == 0:
        print("No experience data to train on yet.")
        return

    print(f"\nTraining on {len(experiences)} moves across {epochs} epochs...")

    # model.train() switches on training-specific behaviour.
    model.train()

    # Adam is the optimizer - it reads the loss and decides how much to
    # nudge each weight. lr (learning rate) controls the step size.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- Unpack all experiences into three separate tensors ---
    # torch.stack turns a list of tensors into one big 2D tensor.
    # Shape: (num_experiences, 773)
    board_tensors = torch.stack([e[0] for e in experiences]).to(device)

    # Move indices: which move (0–4095) the user played each time.
    # dtype=long because these are used as array indices, not floats.
    move_indices = torch.tensor(
        [e[1] for e in experiences], dtype=torch.long
    ).to(device)

    # Outcomes: +1.0, 0.0, or -1.0 for each experience.
    outcomes = torch.tensor(
        [e[2] for e in experiences], dtype=torch.float32
    ).to(device)

    n = len(experiences)

    for epoch in range(epochs):
        # Shuffle the data each epoch so the model doesn't overfit to order.
        # torch.randperm generates a random ordering of indices 0..n-1.
        perm = torch.randperm(n, device=device)
        board_tensors = board_tensors[perm]
        move_indices  = move_indices[perm]
        outcomes      = outcomes[perm]

        total_loss = 0.0
        num_batches = 0

        # tqdm wraps the range and draws a live progress bar in the terminal.
        batch_range = tqdm(
            range(0, n, batch_size),
            desc=f"  Epoch {epoch + 1}/{epochs}",
            leave=False,
        )

        for i in batch_range:
            # Slice out one batch.
            b_boards   = board_tensors[i : i + batch_size]
            b_moves    = move_indices[i : i + batch_size]
            b_outcomes = outcomes[i : i + batch_size]

            # Forward pass: model returns raw scores for all 4096 moves.
            logits = model(b_boards)  # shape: (batch_size, 4096)

            # log_softmax returns log-probabilities.
            # We use log-probabilities because they are numerically stabler
            # and easier to weight than plain probabilities.
            log_probs = F.log_softmax(logits, dim=1)  # shape: (batch_size, 4096)

            # Pick out only the log-probability of the move that was actually played.
            # range(len(b_boards)) selects the row, b_moves selects the column.
            chosen_log_probs = log_probs[range(len(b_boards)), b_moves]  # shape: (batch_size,)

            # Multiply each log-probability by its outcome, then average.
            # Negating flips it into a loss (optimizers minimize, not maximize).
            #   win  → negative * negative = positive loss → model is penalised for low prob
            #   draw → anything * 0        = 0             → no gradient, no change
            #   loss → negative * positive = negative loss → model is pushed away
            loss = -(chosen_log_probs * b_outcomes).mean()

            # Backpropagation: PyTorch works out how each weight contributed
            # to the loss and stores that information as gradients.
            optimizer.zero_grad()  # clear gradients from the previous batch
            loss.backward()        # compute new gradients
            optimizer.step()       # nudge weights in the direction that lowers loss

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch + 1}/{epochs}  -  avg loss: {avg_loss:.4f}")

    print("Training complete.\n")


# ---------------------------------------------------------------------------
# Model persistence - saving and loading weights between sessions
# ---------------------------------------------------------------------------

def save_model(model: ChessNet) -> None:
    """
    Saves the model's weights to disk.
    state_dict() returns a dictionary of all the learned weights.
    We save just the weights, not the model structure (that lives in model.py).
    """
    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def load_model(model: ChessNet, device: torch.device) -> ChessNet:
    """
    Loads previously saved weights into the model.
    If no saved weights exist, the model keeps its random initialisation.
    map_location ensures weights load onto the right device (CPU or GPU).
    """
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        print("Loaded saved model weights.")
    else:
        print("No saved model found - starting fresh with random weights.")
    return model
