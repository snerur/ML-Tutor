"""
Page 8: LLM Analysis
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import utils.compat  # noqa: F401

import streamlit as st
import json

from utils.llm_utils import (
    call_llm,
    test_connection,
    build_system_context,
    PROVIDER_MODELS,
)

st.set_page_config(page_title="LLM Analysis - ML Fairness Studio", layout="wide")

st.title("🤖 LLM Analysis")
st.markdown("Use AI to gain insights about your model's performance, fairness, and feature importance.")

# ── Initialize LLM config in session state ────────────────────────────────────
if "llm_config" not in st.session_state:
    st.session_state["llm_config"] = {
        "provider": "Anthropic (Claude)",
        "model": "claude-sonnet-4-6",
        "api_key": "",
        "ollama_host": "http://localhost:11434",
        "custom_base_url": "http://localhost:11434/v1",
        "custom_model": "",
    }

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ── Sidebar: LLM Configuration ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ LLM Configuration")

    provider_list = list(PROVIDER_MODELS.keys())
    current_provider = st.session_state["llm_config"].get("provider", provider_list[0])
    if current_provider not in provider_list:
        current_provider = provider_list[0]

    selected_provider = st.selectbox(
        "Provider",
        provider_list,
        index=provider_list.index(current_provider),
        key="llm_provider_select",
        help="Choose a cloud/local LLM provider, or 'Custom' for any OpenAI-compatible endpoint.",
    )

    # Model selection — dropdown + free-text override
    models_for_provider = PROVIDER_MODELS.get(selected_provider, [])
    is_custom_provider = selected_provider == "Custom (OpenAI-Compatible)"

    # Always offer a free-text field so user can type any model name
    saved_model = st.session_state["llm_config"].get("model", "")
    saved_custom_model = st.session_state["llm_config"].get("custom_model", "")

    if not is_custom_provider and models_for_provider:
        dropdown_val = saved_model if saved_model in models_for_provider else models_for_provider[0]
        selected_model_dropdown = st.selectbox(
            "Model (preset)",
            models_for_provider,
            index=models_for_provider.index(dropdown_val),
            key="llm_model_select",
        )
        custom_model_name = st.text_input(
            "Or enter a custom model name (overrides dropdown if filled)",
            value=saved_custom_model,
            key="llm_custom_model",
            placeholder="e.g. gpt-4o-2024-08-06 or llama3.2:latest",
        )
        selected_model = custom_model_name.strip() if custom_model_name.strip() else selected_model_dropdown
    else:
        selected_model_dropdown = ""
        custom_model_name = st.text_input(
            "Model name",
            value=saved_custom_model or saved_model,
            key="llm_custom_model",
            placeholder="e.g. my-model, llama3, mistral-7b-instruct",
        )
        selected_model = custom_model_name.strip()

    # API key (not needed for Ollama or key-free custom endpoints)
    needs_key = selected_provider not in ("Ollama (Local)",)
    if needs_key:
        api_key = st.text_input(
            "API Key",
            value=st.session_state["llm_config"].get("api_key", ""),
            type="password",
            key="llm_api_key",
            help=f"Your {selected_provider} API key (leave blank if not required)",
        )
    else:
        api_key = ""

    # Ollama host
    if selected_provider == "Ollama (Local)":
        ollama_host = st.text_input(
            "Ollama Host URL",
            value=st.session_state["llm_config"].get("ollama_host", "http://localhost:11434"),
            key="llm_ollama_host",
        )
    else:
        ollama_host = st.session_state["llm_config"].get("ollama_host", "http://localhost:11434")

    # Custom base URL for OpenAI-compatible endpoints
    if is_custom_provider:
        custom_base_url = st.text_input(
            "Base URL",
            value=st.session_state["llm_config"].get("custom_base_url", "http://localhost:11434/v1"),
            key="llm_custom_base_url",
            help="OpenAI-compatible endpoint (LM Studio, vLLM, Together AI, Mistral, Anyscale, etc.)",
        )
        st.caption("Compatible with LM Studio, vLLM, Together AI, Anyscale, Mistral API, and any server that follows the OpenAI chat completions spec.")
    else:
        custom_base_url = st.session_state["llm_config"].get("custom_base_url", "http://localhost:11434/v1")

    # Generation settings
    st.markdown("### Generation Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05, key="llm_temperature")
    max_tokens = st.slider("Max tokens", 256, 4096, 1500, 128, key="llm_max_tokens")

    # Persist config
    st.session_state["llm_config"] = {
        "provider": selected_provider,
        "model": selected_model,
        "api_key": api_key,
        "ollama_host": ollama_host,
        "custom_base_url": custom_base_url,
        "custom_model": custom_model_name.strip() if 'custom_model_name' in dir() else "",
    }

    # Test connection
    st.markdown("---")
    if st.button("🔌 Test Connection", use_container_width=True):
        with st.spinner("Testing connection..."):
            success, msg = test_connection(
                provider=selected_provider,
                model=selected_model,
                api_key=api_key,
                ollama_host=ollama_host,
                custom_base_url=custom_base_url,
            )
        if success:
            st.success(f"✅ {msg[:80]}")
        else:
            st.error(f"❌ {msg[:150]}")

    st.markdown("---")
    # Clear history
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()

    # Pipeline status summary
    st.markdown("---")
    st.markdown("### Pipeline Status")
    has_data = st.session_state.get("df") is not None
    has_model = st.session_state.get("model") is not None
    has_results = bool(st.session_state.get("test_results"))
    has_fairness = bool(st.session_state.get("fairness_results"))

    st.markdown(f"{'✅' if has_data else '⬜'} Data Loaded")
    st.markdown(f"{'✅' if has_model else '⬜'} Model Trained")
    st.markdown(f"{'✅' if has_results else '⬜'} Test Evaluated")
    st.markdown(f"{'✅' if has_fairness else '⬜'} Fairness Computed")


# ── System context ────────────────────────────────────────────────────────────
system_context = build_system_context(st.session_state)

# ── Pre-built Prompts ─────────────────────────────────────────────────────────
st.markdown("## 💡 Quick Analysis Prompts")
st.markdown("Click a button to ask the LLM a pre-defined question about your ML pipeline:")

prompt_buttons = [
    ("📈 Analyze Model Performance",
     "Analyze the model's performance metrics including accuracy, AUC-ROC, and per-class metrics. "
     "Identify strengths and weaknesses. What do the metrics tell us about the model's behavior?"),
    ("⚖️ Explain Fairness Results",
     "Explain the fairness evaluation results in plain language. "
     "Which protected groups are affected? Are the fairness metric values concerning? "
     "What do demographic parity difference, equalized odds, and disparate impact mean in this context?"),
    ("🔧 Suggest Improvements",
     "Based on the model performance and fairness metrics, suggest specific actionable improvements. "
     "Consider preprocessing, model selection, hyperparameter tuning, and bias mitigation strategies."),
    ("🔮 Explain Feature Importance",
     "Explain what the most important features tell us about the model's decision-making. "
     "Are any of the top features proxy variables for protected attributes? "
     "What are the implications for fairness and model interpretability?"),
    ("🚨 Identify Bias Risks",
     "Identify potential sources of bias in this ML pipeline. "
     "Consider data collection bias, historical bias, representation bias, and measurement bias. "
     "What are the risks if this model is deployed, especially regarding protected groups?"),
    ("📄 Generate Summary Report",
     "Generate a comprehensive executive summary of this ML fairness analysis. "
     "Include: dataset description, model performance, fairness assessment, key risks, "
     "and recommendations for responsible deployment. Format it as a professional report."),
]

# Display buttons in 2 columns
cols_row = st.columns(3)
for i, (btn_label, prompt_text) in enumerate(prompt_buttons):
    with cols_row[i % 3]:
        if st.button(btn_label, use_container_width=True, key=f"prompt_btn_{i}"):
            # Add user message to history
            st.session_state["chat_history"].append({
                "role": "user",
                "content": prompt_text,
            })
            # Trigger LLM call (handled below)
            st.session_state["_trigger_llm"] = True
            st.rerun()

# ── Chat Interface ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 💬 Chat")

# Display chat history
for message in st.session_state["chat_history"]:
    role = message["role"]
    content = message["content"]
    with st.chat_message(role):
        st.markdown(content)

# Handle LLM response if triggered by button or new user input
def get_llm_response():
    """Call LLM with current chat history and system context."""
    messages = [{"role": "system", "content": system_context}]
    messages.extend(st.session_state["chat_history"])

    try:
        cfg = st.session_state["llm_config"]
        response = call_llm(
            provider=cfg["provider"],
            model=cfg["model"],
            api_key=cfg.get("api_key", ""),
            messages=messages,
            temperature=st.session_state.get("llm_temperature", 0.7),
            max_tokens=st.session_state.get("llm_max_tokens", 1500),
            ollama_host=cfg.get("ollama_host", "http://localhost:11434"),
            custom_base_url=cfg.get("custom_base_url", "http://localhost:11434/v1"),
        )
        return response, None
    except Exception as e:
        return None, str(e)


# Auto-call LLM if triggered by button
if st.session_state.pop("_trigger_llm", False):
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, error = get_llm_response()
        if error:
            st.error(f"❌ LLM Error: {error}")
            st.markdown("**Troubleshooting tips:**")
            st.markdown("- Check your API key in the sidebar")
            st.markdown("- Verify the provider and model selection")
            st.markdown("- For Ollama, ensure the server is running at the specified host")
        else:
            st.markdown(response)
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": response,
            })

# Chat input
user_input = st.chat_input("Ask anything about your ML pipeline, fairness, or model...")

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Add to history
    st.session_state["chat_history"].append({
        "role": "user",
        "content": user_input,
    })

    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, error = get_llm_response()

        if error:
            error_msg = f"❌ LLM Error: {error}\n\n**Troubleshooting:**\n- Check API key in sidebar\n- Verify provider/model selection"
            st.error(error_msg)
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": f"Error: {error}",
            })
        else:
            st.markdown(response)
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": response,
            })

# ── Context Preview ───────────────────────────────────────────────────────────
with st.expander("🔍 View System Context (what the LLM sees)", expanded=False):
    st.markdown("The following context is automatically built from your pipeline state and sent to the LLM:")
    st.code(system_context, language="text")

# ── Setup hints ───────────────────────────────────────────────────────────────
if not has_data and not has_model:
    st.info("""
    💡 **Getting started with LLM Analysis:**
    1. Complete the ML pipeline first (upload data → preprocess → train model → evaluate)
    2. Configure your LLM provider in the sidebar (API key required for cloud providers)
    3. Use the quick prompts above or ask your own questions
    4. The LLM will have context about your specific model, data, and fairness results
    """)
elif not has_model:
    st.info("💡 Train a model to get richer LLM analysis. The LLM can already analyze your dataset!")
