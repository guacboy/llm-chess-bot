export default function Status({ text }) {
    if (!text) return null;
    return (
        <div style={styles.box}>
            {text}
        </div>
    );
}

const styles = {
    box: {
        padding: "10px 12px",
        background: "#16213e",
        border: "1px solid #333",
        borderRadius: 4,
        fontSize: 13,
        lineHeight: "1.6",
        color: "#ccc",
    },
};
