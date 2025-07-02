import ollama
import re

def aspect_based_sentiment_llm(review_text):
    """
    Use a local LLM (e.g., phi3:mini or mistral:instruct) to extract aspects and their sentiment
    from a review. Output is a Python dict like your old code: {aspect: sentiment}.
    """
    prompt = f"""
You're a sentiment analysis expert.

Analyze this product review:
"{review_text}"

Extract important product aspects and label their sentiment as Positive, Negative, or Neutral.

Only return each aspect and sentiment as:
aspect1: sentiment
aspect2: sentiment
...

Don't include any explanation or formatting — just aspect: sentiment pairs.
    """

    try:
        response = ollama.chat(
            model='llama3.2',  # or mistral:instruct if more capable
            messages=[{"role": "user", "content": prompt}]
        )
        lines = response['message']['content'].strip().splitlines()
        aspect_sentiment = {}

        for line in lines:
            match = re.match(r"(.+?):\s*(Positive|Negative|Neutral)", line, re.IGNORECASE)
            if match:
                aspect = match.group(1).strip().lower()
                sentiment = match.group(2).capitalize()
                aspect_sentiment[aspect] = sentiment

        return aspect_sentiment

    except Exception as e:
        print("ABSA LLM error:", e)
        return {}
