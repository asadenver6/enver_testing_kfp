import os
import streamlit as st
import vertexai
from vertexai import agent_engines

# ── Config ──
PROJECT_ID = "rta-genai-explorations-406d"
REGION = "europe-west1"
RESOURCE_NAME = "projects/856095474659/locations/europe-west1/reasoningEngines/6904502013875191808"

st.set_page_config(page_title="RFID Knowledge Assistant", page_icon="📡")
st.title("📡 RFID Knowledge Assistant")

# ── Connect to deployed agent (cached so it only connects once per session) ──
@st.cache_resource
def get_agent():
    vertexai.init(project=PROJECT_ID, location=REGION)
    return agent_engines.get(RESOURCE_NAME)

agent = get_agent()

# ── Chat history ──
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ──
if prompt := st.chat_input("Ask about RFID..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        for event in agent.stream_query(user_id="streamlit-user", message=prompt):
            # Extract just the text parts from the event stream
            if "content" in event and "parts" in event["content"]:
                for part in event["content"]["parts"]:
                    if "text" in part:
                        full_response += part["text"]
                        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})