"""
ai_helper.py
------------
Handles all AI-powered features: dataset insights and chat interface.
Uses the Anthropic Claude API (or OpenAI — switchable).
"""

import os
import anthropic


def _get_client():
    """
    Initializes and returns the Anthropic client.
    Reads the API key from the ANTHROPIC_API_KEY environment variable.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Please add it to your .env file or export it in your shell."
        )
    return anthropic.Anthropic(api_key=api_key)


def generate_insights(data_summary: str) -> str:
    """
    Sends the dataset summary to Claude and returns AI-generated insights
    in plain, beginner-friendly English.

    Args:
        data_summary: A text summary of the dataset (from build_summary_text).

    Returns:
        A string with key insights, trends, and anomaly warnings.
    """
    client = _get_client()

    system_prompt = """You are a senior data analyst explaining findings to a non-technical audience.
Your job is to:
1. Summarize what the dataset is about
2. Highlight interesting patterns or trends
3. Flag potential data quality issues (missing values, outliers)
4. Point out 2-3 actionable insights
5. Keep it clear, concise, and jargon-free

Format your response with clear sections using emojis as headers.
Keep total response under 400 words."""

    user_prompt = f"""Analyze this dataset summary and provide key insights in simple, plain English:

{data_summary}

Structure your response as:
📌 **What This Dataset Is About**
📈 **Key Trends & Patterns**
⚠️ **Data Quality Issues**
💡 **Top Insights & Recommendations**"""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=600,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return message.content[0].text


def chat_with_data(question: str, data_summary: str, chat_history: list) -> str:
    """
    Answers a user's question about their dataset using the AI.

    Args:
        question:     The user's question (string).
        data_summary: Text summary of the dataset.
        chat_history: List of {"role": "user/assistant", "content": "..."} dicts.

    Returns:
        AI's answer as a string.
    """
    client = _get_client()

    system_prompt = f"""You are a friendly data analyst assistant.
The user has uploaded a CSV dataset. Here is the dataset summary for context:

{data_summary}

Answer the user's questions about this dataset accurately and concisely.
- If you cannot determine something from the summary alone, say so honestly.
- Use simple language. Avoid jargon unless the user seems technical.
- Keep answers focused and under 200 words unless a detailed explanation is needed.
- If asked for calculations you can't do, suggest what the user should look for."""

    # Build message history for multi-turn conversation
    messages = []
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=400,
        system=system_prompt,
        messages=messages,
    )

    return response.content[0].text


def generate_column_description(column_name: str, sample_values: list, dtype: str) -> str:
    """
    Generates a one-line plain-English description of what a column likely represents.
    Useful for auto-labeling columns with cryptic names.

    Args:
        column_name:   The column name.
        sample_values: A few sample values from the column.
        dtype:         The data type (int, float, object, etc.).

    Returns:
        A one-sentence description string.
    """
    client = _get_client()

    prompt = f"""Given a CSV column named '{column_name}' with data type '{dtype}'
and sample values: {sample_values[:5]},
write ONE concise sentence describing what this column likely represents.
Be specific. No filler phrases."""

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=80,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text.strip()
