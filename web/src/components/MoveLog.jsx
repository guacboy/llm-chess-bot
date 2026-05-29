import { useEffect, useRef } from "react";

export default function MoveLog({ moves }) {
    const bottomRef = useRef(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [moves]);

    return (
        <div style={styles.container}>
            <div style={styles.header}>Move Log</div>
            <div style={styles.list}>
                {moves.length === 0 && (
                    <div style={styles.empty}>No moves yet.</div>
                )}
                {moves.map((m, i) => (
                    <div
                        key={i}
                        style={{
                            ...styles.entry,
                            color: m.by === "bot" ? "#c9a84c" : "#eee",
                        }}
                    >
                        {m.text}
                    </div>
                ))}
                <div ref={bottomRef} />
            </div>
        </div>
    );
}

const styles = {
    container: {
        marginTop: 16,
        border: "1px solid #333",
        borderRadius: 4,
        overflow: "hidden",
    },
    header: {
        padding: "6px 10px",
        background: "#16213e",
        fontSize: 11,
        letterSpacing: 1,
        textTransform: "uppercase",
        color: "#888",
    },
    list: {
        maxHeight: 320,
        overflowY: "auto",
        padding: "6px 10px",
        background: "#0f0f1a",
    },
    entry: {
        fontSize: 13,
        lineHeight: "1.8",
    },
    empty: {
        fontSize: 13,
        color: "#555",
    },
};
