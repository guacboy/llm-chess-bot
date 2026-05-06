import chess
import numpy as np
import torch

# The total number of numbers we use to describe one board position.
# The model will use this as its input size.
INPUT_SIZE = 773


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Converts a chess board into a 773-element tensor of floats.

    Layout:
      [0 : 768]  12 piece planes (6 piece types x 2 colors), each 8x8, flattened
      [768]      whose turn it is  (1.0 = white, 0.0 = black)
      [769: 773] castling rights   (white kingside, white queenside,
                                    black kingside, black queenside)
    """

    # Step 1 - Create 12 empty 8x8 grids, one per piece type per color.
    # Shape: (12 planes, 8 rows, 8 columns), all zeros to start.
    planes = np.zeros((12, 8, 8), dtype=np.float32)

    # Step 2 - Loop over every piece currently on the board and mark its square.
    # board.piece_map() returns a dict of {square_index: Piece}.
    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)   # row 0-7  (rank 1-8)
        file = chess.square_file(square)   # column 0-7  (file a-h)

        # piece.piece_type is 1-6: PAWN=1, KNIGHT=2, BISHOP=3,
        #                          ROOK=4,  QUEEN=5,  KING=6
        # Subtract 1 to get a 0-based index into our 6-plane block.
        piece_idx = piece.piece_type - 1

        # White pieces live in planes 0-5, black pieces in planes 6-11.
        color_offset = 0 if piece.color == chess.WHITE else 6

        # Given the right sheet, the right row, the right column: write 1.0
        # - meaning "yes, there is a piece here."
        planes[color_offset + piece_idx, rank, file] = 1.0

    # Step 3 - Flatten the 12 planes from shape (12, 8, 8) into 768 numbers.
    # (like unrolling the 12 sheets into one long strip)
    flat_planes = planes.flatten()

    # Step 4 - Encode whose turn it is as a single number.
    side_to_move = np.array(
        [1.0 if board.turn == chess.WHITE else 0.0],
        dtype=np.float32
    )

    # Step 5 - Encode the four castling rights as four numbers (1.0 or 0.0).
    castling = np.array([
        float(board.has_kingside_castling_rights(chess.WHITE)),
        float(board.has_queenside_castling_rights(chess.WHITE)),
        float(board.has_kingside_castling_rights(chess.BLACK)),
        float(board.has_queenside_castling_rights(chess.BLACK)),
    ], dtype=np.float32)

    # Step 6 - Concatenate everything into one flat 773-element vector,
    # then wrap it in a PyTorch tensor so the model can use it.
    encoded = np.concatenate([flat_planes, side_to_move, castling])
    return torch.tensor(encoded, dtype=torch.float32)
