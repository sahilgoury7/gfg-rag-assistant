"""GFG RAG Assistant — Streamlit Chat Interface.

Modern, interactive conversational UI for querying 3.4 Lakh GeeksforGeeks
articles indexed in ChromaDB Cloud with grounded Dual-LLM answers and
verifiable citations.
"""

import os
import time
import requests
import streamlit as st

def stream_text(text: str):
    """Yields word tokens with a short delay for a fluid typewriter rendering."""
    words = text.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(0.008)


# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GFG RAG Assistant",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

DEFAULT_CLOUD_BACKEND = os.getenv("BACKEND_URL", "https://gfg-rag-assistant.onrender.com")
DEFAULT_LOCAL_BACKEND = "http://127.0.0.1:8000"

# -----------------------------------------------------------------------------
# 2. Sidebar Configuration & Health Monitoring
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings & Status", divider="gray")

    backend_mode = st.radio(
        "Backend environment",
        options=["Render Cloud", "Localhost", "Custom URL"],
        index=0,
        help="Select which API backend server to query.",
    )

    if backend_mode == "Render Cloud":
        backend_url = DEFAULT_CLOUD_BACKEND
    elif backend_mode == "Localhost":
        backend_url = DEFAULT_LOCAL_BACKEND
    else:
        backend_url = st.text_input("Custom backend URL", value=DEFAULT_CLOUD_BACKEND)

    backend_url = backend_url.rstrip("/")

    # Liveness Probe Function
    def check_health(url: str) -> tuple[bool, str]:
        try:
            resp = requests.get(f"{url}/health", timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                return True, f"v{data.get('version', '1.0')} ({data.get('environment', 'prod')})"
            return False, f"HTTP {resp.status_code}"
        except Exception as e:
            return False, "Offline / Sleep"

    is_healthy, health_detail = check_health(backend_url)

    if is_healthy:
        st.success(f"Backend online: {health_detail}", icon=":material/check_circle:")
    else:
        st.warning(
            f"Backend unreachable: {health_detail}\n\n"
            "*(Render Free Tier may be waking from sleep. Please wait ~30s)*",
            icon=":material/cloud_off:",
        )

    st.subheader("Retrieval parameters")
    top_k = st.slider(
        "Source chunks to retrieve (Top-K)",
        min_value=1,
        max_value=10,
        value=4,
        step=1,
        help="Number of most relevant GFG article chunks to feed into LLM context.",
    )

    st.divider()

    if st.button("Clear chat history", icon=":material/delete:", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "**Architecture:**\n"
        "- 49,298 GFG Articles (3.4L Chunks)\n"
        "- ChromaDB Cloud (HNSW)\n"
        "- ONNX Embeddings (384-d)\n"
        "- Groq LPU + Gemini Failover"
    )

# -----------------------------------------------------------------------------
# 3. Main Header & Hero
# -----------------------------------------------------------------------------
st.title("📚 GeeksforGeeks RAG Assistant")
st.caption(
    "Grounding your Computer Science and DSA questions in **49,298 verified GeeksforGeeks articles (3.4 Lakh chunks)** "
    "with instant source citations and strict context grounding."
)

# -----------------------------------------------------------------------------
# 4. Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------------------------------------------------
# 5. Suggestion Chips (Onboarding experience before first prompt)
# -----------------------------------------------------------------------------
SUGGESTIONS = {
    "💡 What is Dynamic Programming?": "What is Dynamic Programming and what are its key properties?",
    "⚡ Explain Binary Search": "Explain Binary Search with time complexity and iterative code.",
    "🔍 BFS vs DFS in Graphs": "What is the difference between BFS and DFS traversal in graphs?",
    "🚀 Dijkstra's Shortest Path": "How does Dijkstra's Algorithm work with an adjacency list?",
}

if not st.session_state.messages:
    selected_suggestion = st.pills(
        "Try asking:",
        options=list(SUGGESTIONS.keys()),
        label_visibility="collapsed",
    )
    if selected_suggestion:
        st.session_state.messages.append(
            {"role": "user", "content": SUGGESTIONS[selected_suggestion]}
        )
        st.rerun()

# -----------------------------------------------------------------------------
# 6. Render Existing Chat History
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    role = msg["role"]
    avatar = ":material/person:" if role == "user" else ":material/smart_toy:"
    with st.chat_message(role, avatar=avatar):
        st.markdown(msg["content"])

        # Display citations if stored with assistant message
        citations = msg.get("citations", [])
        if citations:
            with st.expander(f"📚 View source articles ({len(citations)} citations)"):
                for idx, c in enumerate(citations, 1):
                    score = c.get("relevance_score", 0.0)
                    pct = int(score * 100)
                    title = c.get("title", "GeeksforGeeks Article")
                    url = c.get("url")

                    if url:
                        st.markdown(f"**{idx}. [{title}]({url})** — `:green[{pct}% match]`")
                    else:
                        st.markdown(f"**{idx}. {title}** — `:green[{pct}% match]`")

        # Display telemetry if available
        telemetry = msg.get("telemetry")
        if telemetry:
            latency_s = telemetry.get("latency_ms", 0.0) / 1000
            provider = telemetry.get("provider", "groq").upper()
            fallback = telemetry.get("fallback", False)
            fb_badge = " | :orange[Fallback used]" if fallback else ""
            st.caption(f"⚡ *Response time: {latency_s:.2f}s | Provider: {provider}{fb_badge}*")

# -----------------------------------------------------------------------------
# 7. Chat Input & Backend Query Execution
# -----------------------------------------------------------------------------
if prompt := st.chat_input("Ask any DSA or Computer Science question..."):
    # 1. Append & display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(prompt)

    # 2. Query the FastAPI Backend
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        with st.status(
            ":shimmer[Searching 3.4 Lakh chunks across 49,298 GFG articles & generating grounded answer...]",
            type="compact",
        ) as status_box:
            start_clock = time.perf_counter()
            error_msg = None
            response_data = None

            try:
                api_response = requests.post(
                    f"{backend_url}/api/v1/query",
                    json={"query": prompt, "top_k": top_k},
                    timeout=90,
                )

                if api_response.status_code == 200:
                    response_data = api_response.json()
                    elapsed = time.perf_counter() - start_clock
                    status_box.update(
                        label=f"Retrieved in {elapsed:.1f}s via {response_data.get('provider_used', 'AI').upper()}",
                        state="complete",
                    )
                elif api_response.status_code == 422:
                    error_msg = "Your query was rejected by input validation rules (e.g. empty or too long)."
                    status_box.update(label="Validation error", state="error")
                elif api_response.status_code == 503:
                    error_msg = "Both Groq and Gemini LLM providers are temporarily unavailable. Please retry in a few moments."
                    status_box.update(label="Service unavailable (503)", state="error")
                else:
                    error_msg = f"Server returned unexpected error: HTTP {api_response.status_code}"
                    status_box.update(label="Server error", state="error")

            except requests.exceptions.Timeout:
                error_msg = "Request timed out. The cloud server may be waking from sleep or experiencing high load. Please retry."
                status_box.update(label="Request timeout", state="error")
            except requests.exceptions.ConnectionError:
                error_msg = f"Failed to connect to backend at `{backend_url}`. Please verify the server is running."
                status_box.update(label="Connection failed", state="error")
            except Exception as e:
                error_msg = f"An unexpected error occurred: {str(e)}"
                status_box.update(label="Execution error", state="error")

        # 3. Render Assistant Response & Citations
        if response_data:
            answer_text = response_data.get("answer", "")
            citations = response_data.get("citations", [])
            telemetry = {
                "latency_ms": response_data.get("latency_ms", 0.0),
                "provider": response_data.get("provider_used", "groq"),
                "fallback": response_data.get("fallback_triggered", False),
            }

            st.write_stream(stream_text(answer_text))

            if citations:
                with st.expander(f"📚 View source articles ({len(citations)} citations)"):
                    for idx, c in enumerate(citations, 1):
                        score = c.get("relevance_score", 0.0)
                        pct = int(score * 100)
                        title = c.get("title", "GeeksforGeeks Article")
                        url = c.get("url")

                        if url:
                            st.markdown(f"**{idx}. [{title}]({url})** — `:green[{pct}% match]`")
                        else:
                            st.markdown(f"**{idx}. {title}** — `:green[{pct}% match]`")

            latency_s = telemetry["latency_ms"] / 1000
            provider_tag = telemetry["provider"].upper()
            fb_badge = " | :orange[Fallback used]" if telemetry["fallback"] else ""
            st.caption(f"⚡ *Response time: {latency_s:.2f}s | Provider: {provider_tag}{fb_badge}*")

            # Store in session state for rerun persistence
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "citations": citations,
                    "telemetry": telemetry,
                }
            )

        elif error_msg:
            st.error(error_msg, icon=":material/error:")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"⚠️ **Error:** {error_msg}"}
            )
