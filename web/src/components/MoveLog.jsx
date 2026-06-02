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
        border: "1px solid rgba(142,175,212,0.15)",
        borderRadius: 6,
        overflow: "hidden",
    },
    header: {
        padding: "6px 10px",
        background: "rgba(255,255,255,0.03)",
        fontSize: 11,
        letterSpacing: 2,
        textTransform: "uppercase",
        color: "#4a5a6a",
        fontFamily: "'Rajdhani', sans-serif",
    },
    list: {
        maxHeight: 320,
        overflowY: "auto",
        padding: "6px 10px",
        background: "rgba(0,0,0,0.4)",
    },
    entry: {
        fontSize: 13,
        lineHeight: "1.8",
        fontFamily: "'Share Tech Mono', monospace",
    },
    empty: {
        fontSize: 13,
        color: "#333a44",
        fontFamily: "'Share Tech Mono', monospace",
    },
};
