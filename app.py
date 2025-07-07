import streamlit as st
# Page Configuration
st.set_page_config(
    page_title="RBI Document Assistant", 
    layout="wide"  # Change "centered" to "wide"
)

from ai import RBIChatbotWithContext
import json
from datetime import datetime
import base64
import os

@st.cache_data
def get_base64_image(image_path):
    """Convert an image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

img_path = os.path.join(os.path.dirname(__file__), "img.png")
img = get_base64_image(img_path)
    
# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = RBIChatbotWithContext(max_context_pairs=5)
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Header

# Gradient box for the title and subtitle
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #c94b27 0%, #f7931e 100%);
        border-radius: 24px;
        padding: 36px 20px 24px 20px;
        margin-bottom: 28px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        color: white;
    ">
        <div style="font-size:2.3rem; font-weight:700; font-family: 'Segoe UI', 'Arial', sans-serif; letter-spacing:0.5px;">
            RBI Document Assistant
        </div>
        <div style="font-size:1.25rem; font-weight:400; margin-top:18px; font-family: 'Segoe UI', 'Arial', sans-serif;">
            Ask questions about RBI guidelines and regulations
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Chat Interface
chat_container = st.container()

with chat_container:
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                st.write(message["content"])
                
                # Show context/sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Sources", expanded=False):
                        for i, source in enumerate(message["sources"], 1):
                            st.caption(f"{i}. {source}")

# Chat input
if prompt := st.chat_input("Ask about RBI guidelines..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Get response from chatbot
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            result = st.session_state.chatbot.ask_question(prompt, k=4)
            
            # Display response
            st.write(result['answer'])
            
            # Show sources
            if result['sources']:
                with st.expander("📚 Sources", expanded=False):
                    for i, source in enumerate(result['sources'], 1):
                        st.caption(f"{i}. {source}")
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result['answer'],
                "sources": result['sources']
            })

# Sidebar with minimal controls
with st.sidebar:
    st.header("Controls")
    
    if st.button("Clear Chat", type="secondary"):
        st.session_state.messages = []
        st.session_state.chatbot.clear_conversation()
        st.rerun()
    
    # Context info
    context_summary = st.session_state.chatbot.get_context_summary()
    st.caption(f"Conversations: {context_summary['total_conversations']}")
    st.caption(f"Context length: {context_summary['recent_context_pairs']}")
    
    # Save conversation
    if st.button("Save Chat", type="secondary"):
        if st.session_state.messages:
            filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            chat_data = {
                "messages": st.session_state.messages,
                "timestamp": datetime.now().isoformat()
            }
            st.download_button(
                label="Download Chat",
                data=json.dumps(chat_data, indent=2),
                file_name=filename,
                mime="application/json"
            )
            
# Sidebar background
# sidebar_bg = f"""
#     <style>
#     [data-testid="stSidebar"] {{
#         background-image: url("data:image/png;base64,{img}");
#         background-size: cover;
#         background-position: center;
#     }}
#     </style>
# """
# st.markdown(sidebar_bg, unsafe_allow_html=True)

# Main app background 
main_bg = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{img}");
        background-size: cover;
        background-position: center;
    }}
    [data-testid="stHeader"] {{
        background-image: url("data:image/png;base64,{img}");
        background-size: cover;
        background-position: center;
    }}
    [data-testid="stBottomBlockContainer"] {{
        background-image: url("data:image/png;base64,{img}");
        background-size: cover;
        background-position: center;
    }}
    [data-testid="stBottom"]{{
    bacgroungd-image: url("data:image/png;base64,{img}");
    background-size: cover;
    background-position: center;
    }}
    </style>
"""
st.markdown(main_bg, unsafe_allow_html=True)