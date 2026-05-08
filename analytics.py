def build_dashboard_payload(get_db_connection, user_id):
    """Build dashboard cards + chart data from prediction history."""
    with get_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT prediction, confidence, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY created_at DESC
            """,
            (user_id,),
        ).fetchall()

    total = len(rows)
    fake_count = sum(1 for row in rows if row["prediction"] == "FAKE")
    real_count = sum(1 for row in rows if row["prediction"] == "REAL")
    avg_confidence = round(sum(row["confidence"] for row in rows) / total, 2) if total else 0

    day_counts = {}
    for row in rows:
        day = row["created_at"][:10]
        day_counts[day] = day_counts.get(day, 0) + 1

    return {
        "total_predictions": total,
        "fake_count": fake_count,
        "real_count": real_count,
        "avg_confidence": avg_confidence,
        "line_labels": list(day_counts.keys())[-10:],
        "line_values": list(day_counts.values())[-10:],
        "recent_rows": rows[:10],
    }
