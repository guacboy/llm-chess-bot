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
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(142,175,212,0.2)",
        borderRadius: 6,
        fontSize: 14,
        lineHeight: "1.6",
        color: "#8eafd4",
        fontFamily: "'Rajdhani', sans-serif",
        letterSpacing: 0.5,
    },
};
