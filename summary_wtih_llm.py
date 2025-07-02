import ollama

def generate_summary(text):
    """
    Generate a short keyphrase-style summary using Phi-3 Mini via Ollama.
    """
    prompt = (
        "Summarize the following product review in **5 to 10 words**, like a quick headline or keyword list.\n"
        "Avoid full sentences. Focus only on the most important points.\n\n"
        f"Review:\n{text}\n\nSummary (short phrases):"
    )

    response = ollama.chat(
        model='llama3.2',
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    summary = response['message']['content'].strip()
    return summary
