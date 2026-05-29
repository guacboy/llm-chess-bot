import { useState, useRef, useCallback } from "react";
import { Chess } from "chess.js";
import Board from "./components/Board";
import MoveLog from "./components/MoveLog";
import Status from "./components/Status";

const WS_URL = import.meta.env.PROD
    ? `ws://${window.location.host}/ws`
    : "ws://localhost:8000/ws";

// TODO(fix): website auto-scrolls down when new move is printed.
// TODO(feat): add draw/abort options, also ability to view previous moves.
// TODO(feat): user's w/l-stats can be viewed at the bottom.
// TODO(feat): settings option (ability to reset model, import games, board coloring?).
// TODO(chore): change background coloring to a more darker theme.
// TODO(chore): center board and options.

export default function App() {
    // phase controls which screen is shown:
    //   "setup"   -> color picker before the game starts.
    //   "playing" -> active game, board accepts moves.
    //   "over"    -> game finished, board is locked, Play Again button appears.
    const [phase, setPhase] = useState("setup");

    // Renders the chessboard starting position.
    const [fen, setFen] = useState("start");

    const [userColor, setUserColor] = useState("white");
    const [moveLog, setMoveLog] = useState([]);
    const [status, setStatus] = useState("");

    const wsRef = useRef(null);

    // Used to validate moves client-side before sending them,
    // so illegal moves are rejected immediately without a round-trip to the server.
    const gameRef = useRef(new Chess());

    // Called when the user clicks "Play as White/Black".
    // Opens the WebSocket connection and registers all message handlers.
    const startGame = useCallback((color) => {
        setUserColor(color);

        const ws = new WebSocket(WS_URL);
        wsRef.current = ws;

        // Once the connection is open, tell the server which color we want.
        ws.onopen = () => {
            ws.send(JSON.stringify({ type: "start", color }));
        };

        // Handle every message the server sends back.
        ws.onmessage = (e) => {
            const msg = JSON.parse(e.data);

            // Server confirmed the game started; reset local state and go to playing phase.
            if (msg.type === "game_started") {
                gameRef.current = new Chess();
                setFen(msg.fen);
                setMoveLog([]);
                setStatus(
                    `Game started - you are ${color}. Bot mirrors your style ${msg.model_pct}% of the time.`
                );
                setPhase("playing");
            }

            // Server acknowledged our move; update the board to the new position.
            if (msg.type === "user_move_ack") {
                gameRef.current.load(msg.fen);
                setFen(msg.fen);
                setMoveLog((prev) => [...prev, { by: "you", text: msg.description }]);
            }

            // Server sent the bot's reply move; update the board again.
            if (msg.type === "bot_move") {
                gameRef.current.load(msg.fen);
                setFen(msg.fen);
                setMoveLog((prev) => [
                    ...prev,
                    { by: "bot", text: `${msg.description}  [${msg.source}]` },
                ]);
            }

            // Game ended: show the result and a breakdown of how the bot played.
            if (msg.type === "game_over") {
                const labels = { win: "You won!", lose: "You lost.", draw: "Draw." };
                const { model: m = 0, stockfish: sf = 0, random: r = 0 } = msg.move_counts;
                const total = m + sf + r;
                const mPct = total > 0 ? Math.round((m / total) * 100) : 0;
                const sfPct = total > 0 ? Math.round((sf / total) * 100) : 0;
                setStatus(
                    `${labels[msg.result] ?? msg.result}  ·  Bot used your style ${mPct}%, Stockfish ${sfPct}%`
                );
                setPhase("over");
            }

            // Training finished in the background after the game; append a note to status.
            if (msg.type === "training_complete") {
                setStatus((prev) => prev + "  ·  Model updated.");
            }

            if (msg.type === "error") {
                console.warn("Server error:", msg.message);
            }
        };

        ws.onerror = () => setStatus("Connection error - is the server running?");
    }, []);

    // Called by the Board component when the user drags a piece.
    // Returns true to confirm the move visually, false to snap the piece back.
    const handleMove = useCallback((sourceSquare, targetSquare) => {
        // Check legality locally before sending to the server.
        const moves = gameRef.current.moves({ verbose: true });
        const match = moves.find(
            (m) => m.from === sourceSquare && m.to === targetSquare
        );
        if (!match) return false;

        // Apply the move immediately so the board updates visually right away.
        gameRef.current.move(match);
        setFen(gameRef.current.fen());

        // Always promote to queen.
        const uci = match.promotion
            ? `${sourceSquare}${targetSquare}q`
            : `${sourceSquare}${targetSquare}`;

        wsRef.current?.send(JSON.stringify({ type: "move", uci }));
        return true;
    }, []);

    // Resets everything back to the color-picker screen.
    const handlePlayAgain = () => {
        wsRef.current?.close();
        setPhase("setup");
        setFen("start");
        setMoveLog([]);
        setStatus("");
        gameRef.current = new Chess();
    };

    return (
        <div style={styles.page}>
            <h1 style={styles.title}>Mirror AI Chess Bot</h1>

            {phase === "setup" && (
                <div style={styles.setup}>
                    <p style={styles.subtitle}>Pick your color to start a game.</p>
                    <div style={styles.colorButtons}>
                        <button style={styles.btn} onClick={() => startGame("white")}>
                            Play as White
                        </button>
                        <button style={styles.btn} onClick={() => startGame("black")}>
                            Play as Black
                        </button>
                    </div>
                </div>
            )}

            {phase !== "setup" && (
                <div style={styles.game}>
                    <Board
                        fen={fen}
                        userColor={userColor}
                        onMove={handleMove}
                        disabled={phase === "over"}
                    />
                    <div style={styles.sidebar}>
                        <Status text={status} />
                        <MoveLog moves={moveLog} />
                        {phase === "over" && (
                            <button
                                style={{ ...styles.btn, marginTop: 16 }}
                                onClick={handlePlayAgain}
                            >
                                Play Again
                            </button>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}

const styles = {
    page: {
        minHeight: "100vh",
        background: "#1a1a2e",
        color: "#eee",
        fontFamily: "monospace",
        padding: "24px",
        boxSizing: "border-box",
    },
    title: {
        margin: "0 0 8px",
        fontSize: 22,
        letterSpacing: 1,
        color: "#c9a84c",
    },
    subtitle: {
        margin: "0 0 20px",
        color: "#aaa",
    },
    setup: {
        display: "flex",
        flexDirection: "column",
        alignItems: "flex-start",
        marginTop: 40,
    },
    colorButtons: {
        display: "flex",
        gap: 12,
    },
    btn: {
        padding: "10px 22px",
        background: "#c9a84c",
        color: "#1a1a2e",
        border: "none",
        borderRadius: 4,
        cursor: "pointer",
        fontFamily: "monospace",
        fontWeight: "bold",
        fontSize: 14,
    },
    game: {
        display: "flex",
        gap: 28,
        alignItems: "flex-start",
        marginTop: 20,
        flexWrap: "wrap",
    },
    sidebar: {
        display: "flex",
        flexDirection: "column",
        width: 280,
        minWidth: 220,
    },
};
