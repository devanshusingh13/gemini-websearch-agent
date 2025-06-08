import streamlit as st
import requests
from services.api_client import send_query
from services.db_client import (
    register_user, login_user, get_past_conversations
)

# --- Session State Initialization ---
if "jwt_token" not in st.session_state:
    st.session_state.jwt_token = None
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: Registration ---
with st.sidebar.expander("📝 Register New Account"):
    new_name = st.text_input("Name", key="reg_name")
    new_email = st.text_input("Email", key="reg_email")
    new_password = st.text_input(
        "Password", type="password", key="reg_password")
    if st.button("Register"):
        if new_name and new_email and new_password:
            success, msg = register_user(new_name, new_email, new_password)
            if success:
                st.success(msg)
            else:
                st.error(msg)
        else:
            st.error("Please fill out all fields.")

# --- Sidebar: Login ---
with st.sidebar.expander("🔐 Login"):
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        if email and password:
            token, error = login_user(email, password)
            if token:
                st.session_state.jwt_token = token
                st.session_state.user_email = email
                st.rerun()
            else:
                st.error(error)
        else:
            st.error("Please enter email and password.")

# --- Logout button ---
if st.session_state.jwt_token:
    st.sidebar.success(f"Logged in as {st.session_state.user_email}")
    if st.sidebar.button("Logout"):
        st.session_state.clear()
        st.rerun()

# --- App Title ---
st.title("💬 Gemini-Powered Web Chat")

# --- Sidebar: Past Conversations (with token check) ---
if st.session_state.jwt_token:
    st.sidebar.markdown("### 📜 Past Conversations")
    try:
        past_chats = get_past_conversations(
            st.session_state.jwt_token, st.session_state.user_email
        )
        for convo in past_chats:
            st.sidebar.write(f"🧑 {convo['user_msg']}")
            st.sidebar.write(f"🤖 {convo['ai_msg']}\n")
    except Exception as e:
        st.sidebar.error(f"Failed to load conversations: {e}")

# --- Chat Interface ---
if st.session_state.jwt_token:
    # Show existing message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    user_input = st.chat_input("Type your message...")
    if user_input:
        st.chat_message("user").write(user_input)
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )

        with st.spinner("Gemini is thinking..."):
            try:
                response = send_query(user_input, st.session_state.jwt_token)
                ai_response = response.get("response", "Sorry, no response.")
                st.chat_message("assistant").write(ai_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": ai_response}
                )
            except Exception as e:
                st.error(f"Failed to fetch response: {e}")
else:
    st.info("Please log in to chat.")
