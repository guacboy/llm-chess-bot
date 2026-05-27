import chess
import torch
import torch.nn as nn

INPUT_SIZE = 773    # Total number of numbers we use to describe one board position.
OUTPUT_SIZE = 4096  # 64 * 64 = 4096 possible from/to combinations.


class ChessNet(nn.Module):
    """
    Feedforward neural network that maps a board state to move scores.

    Input:  773 numbers  (the encoded board)
    Output: 4096 numbers (one raw score per possible from/to move)
    """

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(INPUT_SIZE, 512),  # 773 -> 512
            nn.ReLU(),
            nn.Linear(512, 512),         # 512 -> 512
            nn.ReLU(),
            nn.Linear(512, 256),         # 512 -> 256
            nn.ReLU(),
            nn.Linear(256, OUTPUT_SIZE), # 256 -> 4096
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines how data flows through the network.
        """
        return self.network(x)


def encode_move(move: chess.Move) -> int:
    """
    Converts a chess.Move into a single index in the range 0-4095.
    """
    return move.from_square * 64 + move.to_square


def decode_move(index: int) -> tuple[int, int]:
    """
    Converts an index 0-4095 back into (from_square, to_square).
    """
    return index // 64, index % 64


def select_move(
    model: ChessNet,
    board: chess.Board,
    board_tensor: torch.Tensor,
    device: torch.device,
) -> chess.Move:
    """
    Asks the model which move to play and returns a legal chess.Move.

    Steps:
        1. Run the board tensor through the network to get 4096 raw scores.
        2. Set every illegal move's score to -inf so softmax ignores them.
        3. Convert remaining scores to probabilities with softmax.
        4. Return the move with the highest probability.
    """

    # Switches off training-only behaviour.
    model.eval()

    with torch.no_grad():
        # unsqueeze(0) adds a batch dimension: (773,) → (1, 773)
        # squeeze(0)   removes it again after: (1, 4096) → (4096,)
        logits = model(board_tensor.unsqueeze(0).to(device)).squeeze(0)

    # Start with -inf everywhere (every move blocked),
    # then open up the squares that correspond to legal moves.
    mask = torch.full((OUTPUT_SIZE,), float("-inf"))
    legal_moves = list(board.legal_moves)

    for move in legal_moves:
        mask[encode_move(move)] = 0.0

    # Illegal moves become -inf and vanish after softmax.
    probs = torch.softmax(logits + mask.to(device), dim=0)

    # Pick the move index with the highest probability.
    best_idx = int(torch.argmax(probs).item())
    from_sq, to_sq = decode_move(best_idx)

    # Find the matching legal move.
    candidates = [
        m for m in legal_moves
        if m.from_square == from_sq and m.to_square == to_sq
    ]

    for move in candidates:
        if move.promotion == chess.QUEEN:
            return move

    return candidates[0]
