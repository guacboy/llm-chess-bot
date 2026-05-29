import { Chessboard } from "react-chessboard";

// Board renders the interactive chess board.
// Args:
//   fen      : current board position as a FEN string, kept in sync with the server
//   userColor: "white" or "black", determines which side faces the player
//   onMove   : called with (sourceSquare, targetSquare) when a piece is dropped;
//              should return true to confirm the move or false to snap the piece back
//   disabled : when true (game over), drops are ignored so pieces can't be moved
export default function Board({ fen, userColor, onMove, disabled }) {
    // Flips the board so the player's pieces are always at the bottom.
    const orientation = userColor === "white" ? "white" : "black";

    // Returning false snaps the piece back to where it was dragged from.
    const handleDrop = ({ sourceSquare, targetSquare }) => {
        if (disabled) return false;
        return onMove(sourceSquare, targetSquare);
    };

    return (
        // Controls the board's pixel size.
        <div style={{ width: 480, flexShrink: 0 }}>
            {}
            <Chessboard
                options={{
                    position: fen,
                    onPieceDrop: handleDrop,
                    boardOrientation: orientation,
                    darkSquareStyle: { backgroundColor: "#4a7c59" },
                    lightSquareStyle: { backgroundColor: "#f0d9b5" },
                }}
            />
        </div>
    );
}
