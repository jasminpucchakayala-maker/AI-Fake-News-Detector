def summarize_article(article_text):
    """Lightweight summary generator for long text."""
    if not article_text:
        return "Please provide an article to summarize."
    sentences = [sentence.strip() for sentence in article_text.split(".") if sentence.strip()]
    if not sentences:
        return article_text[:180]
    summary = ". ".join(sentences[:2])
    return f"{summary}."


def explain_prediction_heuristic(article_text):
    """Simple explainability heuristic for beginner-friendly interpretation."""
    text = article_text.lower()
    suspicious_terms = ["shocking", "secret", "miracle", "exclusive", "unbelievable", "100% guarantee"]
    matched = [term for term in suspicious_terms if term in text]
    if matched:
        return f"The text contains sensational keywords: {', '.join(matched)}. Such language can indicate misinformation."
    return "The writing style appears relatively neutral. Cross-verify with trusted sources for reliability."


def generate_ai_assistant_response(article_text, user_question):
    """Generate practical misinformation guidance and summary."""
    question = (user_question or "").lower()

    if "summarize" in question:
        return summarize_article(article_text)
    if "why" in question or "fake" in question or "real" in question:
        return explain_prediction_heuristic(article_text)
    if "source" in question:
        return "Check publication reputation, author profile, citations, publication date, and cross-coverage by trusted media."
    if "misinformation" in question or "verify" in question:
        return "Use reverse image search, verify quotes from primary sources, and compare with Reuters/AP/BBC style outlets."

    return (
        "I can summarize this article, explain likely fake-news indicators, and suggest verification steps. "
        "Try asking: 'Summarize this news' or 'Why could this be fake?'."
    )
