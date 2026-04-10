"""
LLM utility functions for ML Fairness Studio.
Unified interface to multiple LLM providers.
"""

import json
import requests

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic as anthropic_lib
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# ── Provider model lists ─────────────────────────────────────────────────────
PROVIDER_MODELS = {
    "OpenAI": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
    ],
    "Anthropic (Claude)": [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
        "claude-opus-4-5",
        "claude-sonnet-4-5",
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
    ],
    "Ollama (Local)": [
        "llama3.2",
        "llama3.1",
        "mistral",
        "mixtral",
        "phi3",
        "gemma2",
        "codellama",
        "qwen2.5",
    ],
    "Groq": [
        "llama-3.1-70b-versatile",
        "llama-3.1-8b-instant",
        "llama3-70b-8192",
        "llama3-8b-8192",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ],
    "HuggingFace": [
        "microsoft/DialoGPT-large",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "HuggingFaceH4/zephyr-7b-beta",
        "tiiuae/falcon-7b-instruct",
    ],
    "Custom (OpenAI-Compatible)": [
        "enter-model-name-below",
    ],
}


def call_llm(provider, model, api_key, messages, temperature=0.7, max_tokens=2048,
             ollama_host="http://localhost:11434", **kwargs):
    """
    Unified LLM call interface.

    Parameters
    ----------
    provider : str
        One of 'OpenAI', 'Anthropic (Claude)', 'Ollama (Local)', 'Groq', 'HuggingFace'
    model : str
        Model name/id
    api_key : str
        API key (not needed for Ollama)
    messages : list of dict
        [{"role": "system"/"user"/"assistant", "content": "..."}]
    temperature : float
    max_tokens : int
    ollama_host : str
        Base URL for Ollama server

    Returns
    -------
    str : The assistant's response text
    """
    if provider == "OpenAI":
        return _call_openai(model, api_key, messages, temperature, max_tokens)
    elif provider == "Anthropic (Claude)":
        return _call_anthropic(model, api_key, messages, temperature, max_tokens)
    elif provider == "Ollama (Local)":
        return _call_ollama(model, messages, temperature, max_tokens, ollama_host)
    elif provider == "Groq":
        return _call_groq(model, api_key, messages, temperature, max_tokens)
    elif provider == "HuggingFace":
        return _call_huggingface(model, api_key, messages, temperature, max_tokens)
    elif provider == "Custom (OpenAI-Compatible)":
        base_url = kwargs.get("custom_base_url", "http://localhost:11434/v1")
        return _call_custom_openai_compatible(model, api_key, messages, temperature, max_tokens, base_url)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _call_openai(model, api_key, messages, temperature, max_tokens):
    if not OPENAI_AVAILABLE:
        raise ImportError("openai package not installed. Run: pip install openai")
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _call_anthropic(model, api_key, messages, temperature, max_tokens):
    if not ANTHROPIC_AVAILABLE:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")

    client = anthropic_lib.Anthropic(api_key=api_key)

    # Separate system message from conversation
    system_prompt = ""
    conversation = []
    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        else:
            conversation.append({"role": msg["role"], "content": msg["content"]})

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": conversation,
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    response = client.messages.create(**kwargs)
    return response.content[0].text


def _call_ollama(model, messages, temperature, max_tokens, ollama_host):
    url = f"{ollama_host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data.get("message", {}).get("content", "")


def _call_groq(model, api_key, messages, temperature, max_tokens):
    if not GROQ_AVAILABLE:
        # Fallback to requests-based approach
        return _call_groq_http(model, api_key, messages, temperature, max_tokens)

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _call_groq_http(model, api_key, messages, temperature, max_tokens):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _call_huggingface(model, api_key, messages, temperature, max_tokens):
    # Use HuggingFace Inference API
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}

    # Build prompt from messages
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"<|system|>\n{content}\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"
    prompt += "<|assistant|>\n"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "return_full_text": False,
        },
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    if isinstance(data, list) and len(data) > 0:
        return data[0].get("generated_text", "")
    return str(data)


def _call_custom_openai_compatible(model, api_key, messages, temperature, max_tokens, base_url):
    """Call any OpenAI-compatible REST endpoint (LM Studio, vLLM, Together, Mistral, etc.)."""
    if not OPENAI_AVAILABLE:
        # Fall back to raw HTTP if openai package missing
        url = f"{base_url.rstrip('/')}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    client = openai.OpenAI(
        api_key=api_key or "not-needed",
        base_url=base_url,
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def test_connection(provider, model, api_key, ollama_host="http://localhost:11434",
                    custom_base_url="http://localhost:11434/v1"):
    """
    Test connectivity to an LLM provider.
    Returns (success: bool, message: str)
    """
    test_messages = [
        {"role": "user", "content": "Reply with exactly: 'Connection successful'"}
    ]
    try:
        response = call_llm(
            provider=provider,
            model=model,
            api_key=api_key,
            messages=test_messages,
            temperature=0.0,
            max_tokens=50,
            ollama_host=ollama_host,
            custom_base_url=custom_base_url,
        )
        return True, f"Connected successfully. Response: {response[:100]}"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"


def ask_ai_recommendation(question, options, session_state, llm_config):
    """
    Ask the LLM for a recommendation when the user faces a choice.

    Parameters
    ----------
    question : str
        The decision the user is trying to make (e.g. "Which ML algorithm should I use?").
    options : list of str
        The available choices.
    session_state : dict
        Current Streamlit session state (used to build pipeline context).
    llm_config : dict
        LLM provider configuration with keys: provider, model, api_key, ollama_host, custom_base_url.

    Returns
    -------
    str : LLM recommendation text, or error message string starting with "Error:".
    """
    context = build_system_context(session_state)
    options_str = "\n".join(f"  - {o}" for o in options)
    prompt = (
        f"{context}\n\n"
        f"The user is deciding: **{question}**\n\n"
        f"Available options:\n{options_str}\n\n"
        "Please recommend the best option(s) for this specific dataset and use case. "
        "Briefly explain your reasoning (2–4 sentences per recommendation). "
        "Be concrete and practical."
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        response = call_llm(
            provider=llm_config.get("provider", "Anthropic (Claude)"),
            model=llm_config.get("model", "claude-sonnet-4-6"),
            api_key=llm_config.get("api_key", ""),
            messages=messages,
            temperature=0.3,
            max_tokens=800,
            ollama_host=llm_config.get("ollama_host", "http://localhost:11434"),
            custom_base_url=llm_config.get("custom_base_url", "http://localhost:11434/v1"),
        )
        return response
    except Exception as exc:
        return f"Error: {exc}"


def build_system_context(session_state):
    """
    Build a system context string from the current session state
    to provide the LLM with relevant information about the ML pipeline.
    """
    lines = ["You are an expert ML fairness analyst. Here is the current state of the ML pipeline:\n"]

    # Dataset info
    df = session_state.get("df")
    if df is not None:
        lines.append(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        target = session_state.get("target_col", "Unknown")
        lines.append(f"Target variable: {target}")
        protected = session_state.get("protected_cols", [])
        if protected:
            lines.append(f"Protected attributes: {', '.join(protected)}")

    # Preprocessing
    pre_config = session_state.get("preprocessing_config", {})
    if pre_config:
        lines.append(f"\nPreprocessing config: {json.dumps(pre_config, indent=2)}")

    X_train = session_state.get("X_train")
    X_test = session_state.get("X_test")
    if X_train is not None:
        lines.append(f"\nTraining set size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    if X_test is not None:
        lines.append(f"Test set size: {X_test.shape[0]} samples")

    # Model info
    model_name = session_state.get("model_name")
    if model_name:
        lines.append(f"\nModel: {model_name}")

    test_results = session_state.get("test_results", {})
    if test_results:
        roc_auc = test_results.get("roc_auc")
        if roc_auc is not None:
            lines.append(f"ROC-AUC: {roc_auc:.4f}")
        report = test_results.get("report", {})
        acc = report.get("accuracy")
        if acc is not None:
            lines.append(f"Accuracy: {acc:.4f}")
        macro = report.get("macro avg", {})
        if macro:
            lines.append(f"Macro F1: {macro.get('f1-score', 'N/A'):.4f}")

    # Fairness metrics
    fairness_results = session_state.get("fairness_results", {})
    if fairness_results:
        lines.append("\nFairness metrics:")
        for k, v in fairness_results.items():
            if v is not None:
                lines.append(f"  {k}: {v}")

    # Feature importance (top 10)
    feature_names = session_state.get("feature_names", [])
    if feature_names:
        lines.append(f"\nNumber of features: {len(feature_names)}")
        lines.append(f"Top features (first 10): {', '.join(feature_names[:10])}")

    lines.append("\nPlease provide expert analysis based on this information.")
    return "\n".join(lines)
