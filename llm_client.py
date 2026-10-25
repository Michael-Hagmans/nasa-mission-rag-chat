"""Grounded answer generation with the OpenAI chat API."""

from typing import Dict, List

import config

SYSTEM_PROMPT = """You answer questions about NASA space missions using only the documents below.

Rules:
- Every statement in your answer must be supported by the documents.
- If the documents do not contain the answer, say so plainly.
- Do not use outside knowledge or fill gaps with plausible-sounding information.
- Do not draw inferences beyond what the documents explicitly state.

Documents:
{context}"""


def generate_response(openai_key: str, user_message: str, context: str,
                      conversation_history: List[Dict], model: str = config.CHAT_MODELS[0]) -> str:
    """Answer user_message from the retrieved context, taking earlier turns into account.

    conversation_history holds the previous user/assistant messages and is not modified.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context or "(no documents retrieved)")},
        *(m for m in conversation_history if m.get("role") in ("user", "assistant")),
        {"role": "user", "content": user_message},
    ]

    response = config.get_openai_client(openai_key).chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=500,
    )
    return response.choices[0].message.content or "No response was generated."
