from io import StringIO
from pathlib import Path
from typing import Dict, List

import streamlit as st

from chatmodel import Chat, new_chat

PREFERRED_MODEL = "phi4:latest"

#
# Helpers
#


def clear_chat():
    """Clears the existing chat session"""
    del st.session_state["chat"]


def _prepare_messages_for_model(messages: List[Dict]):
    """Converts message history format into format for model"""
    # Models like things in this order:
    # - System
    # - Files
    # - Chat
    # System and files for context, chat for the task
    result = []

    system = [m for m in messages if m["role"] == "system"]
    file = [m for m in messages if m["role"] == "file"]
    chat = [m for m in messages if m["role"] not in ["system", "file"]]

    result.extend(system)
    for m in file:
        # Models don't have a file role, so convert
        prompt = f"""
        `{m["name"]}`
        ---
        {m["data"]}

        ---"""
        result.append({"role": "assistant", "content": prompt})
    result.extend(chat)
    return result


def stream_model_response():
    """Returns a generator that yields chunks of the models respose"""
    cg = st.session_state["chat_gateway"]
    response = cg.chat(
        model=st.session_state["chat"].model,
        messages=_prepare_messages_for_model(st.session_state["chat"].messages),
        stream=True,
        num_ctx=min(8192, st.session_state["model_context_length"]),
    )
    for chunk in response:  # prompt eval count is the token count used from the model
        # if chunk.prompt_eval_count is not None:
        #     st.session_state["used_tokens"] = chunk.prompt_eval_count + chunk.eval_count
        yield chunk["message"]["content"]


def regenerate_last_response():
    """Regenerate the last assistant response"""
    # Remove the last assistant message
    st.session_state["chat"].messages = st.session_state["chat"].messages[:-1]
    st.session_state["generate_assistant"] = True


def handle_submit_prompt():
    # add latest message to history in format {role, content}
    prompt = st.session_state["user_prompt"]
    if prompt.startswith("/include"):
        _handle_submit_include(prompt)
    else:
        _handle_submit_chat(prompt)


def _handle_submit_chat(prompt: str):
    """Handle the user submitting a general chat prompt"""
    st.session_state["chat"].messages.append({"role": "user", "content": prompt})
    st.session_state["generate_assistant"] = True


def _handle_submit_include(prompt: str):
    """Handle the user trying to upload a file"""
    # Upload a file to the chat /attach /path/to/file
    parts = prompt.strip().split(" ")
    files = []
    if len(parts) == 2:
        files.append(Path(parts[1]))
    elif len(parts) == 3:  # path and glob
        files.extend(Path(parts[1]).glob(parts[2]))
    print(files)
    for p in files:
        try:
            _insert_file_chat_message(p.read_text(), p.name, p.suffix.lstrip("."))
        except FileNotFoundError:
            print(f"Error: File '{p}' not found.")
            st.toast(f"Error: File '{p}' not found.")
        except PermissionError:
            print(f"Error: Permission denied for accessing the file '{p}'.")
            st.toast(f"Error: Permission denied for accessing the file '{p}'.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            st.toast(f"An unexpected error occurred uploading the file: {e}")


def handle_add_doc_to_chat():
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    string_data = stringio.read()
    ext = uploaded_file.name.split(".")[-1]
    _insert_file_chat_message(string_data, uploaded_file.name, ext)


def _insert_file_chat_message(data, fname, fext: str):
    st.session_state["chat"].messages.append(
        {
            "role": "file",
            "name": fname,
            "ext": fext,
            "data": data,
        }
    )


def handle_change_model():
    model = st.session_state["model"]
    st.session_state["chat"].model = model
    _update_context_length(model)


def _update_context_length(model):
    cg = st.session_state["chat_gateway"]
    m = cg.show(model)
    st.session_state["model_context_length"] = m.context_length


#
# Chat history
#


def generate_chat_title():
    """Generate a title from the first 6 words of the first user message"""
    # Find first user message
    user_messages = [
        msg for msg in st.session_state["chat"].messages if msg["role"] == "user"
    ]
    if not user_messages:
        return "New Chat"

    # Take first 6 words of first message
    first_message = user_messages[0]["content"]
    words = first_message.split()[:10]
    title = " ".join(words)

    # Add ellipsis if we truncated the message
    if len(words) < len(first_message.split()):
        title += "..."

    return title


def save_current_chat():
    """Save the current chat session"""
    if len(st.session_state["chat"].messages) > 1:  # More than just system message
        title = generate_chat_title()
        st.session_state["history_manager"].save_chat(
            st.session_state["chat"].messages,
            title,
            st.session_state["chat"].model,
            st.session_state["chat"].id,
            st.session_state["chat"].created_at,
        )


def load_chat(chat_id):
    """Load a chat from history"""
    chat = st.session_state["history_manager"].get_chat(chat_id)
    if chat:
        st.session_state["chat"] = Chat(
            id=chat_id,
            model=chat["model"],
            messages=chat["messages"],
            created_at=chat["created_at"],
        )


#
# Initialise app state
#


# Don't generate a chat message until the user has prompted
if "generate_assistant" not in st.session_state:
    st.session_state["generate_assistant"] = False

# Store the tokens used in the prompt
if "used_tokens" not in st.session_state:
    st.session_state["used_tokens"] = 0

# Retrieve the current models from ollama
# Set a preferred model as default if there's none set
models = st.session_state["chat_gateway"].list()
if "model" not in st.session_state and PREFERRED_MODEL in models:
    st.session_state["model"] = PREFERRED_MODEL
_update_context_length(st.session_state["model"])

# Start a new chat if there isn't one active.
# "New Chat" is implemented as `del st.session_state["chat"]`
if "chat" not in st.session_state:
    st.session_state["chat"] = new_chat(st.session_state["model"])
    st.session_state["used_tokens"] = 0


#
# Start rendering the app
#


with st.sidebar:
    st.button(
        "New Chat",
        on_click=clear_chat,
        icon=":material/edit_square:",
        use_container_width=True,
    )
    st.selectbox(
        "Choose your model", models, key="model", on_change=handle_change_model
    )

    # Display recent chats
    st.markdown("## Recent Chats")
    recent_chats = st.session_state["history_manager"].get_recent_chats(limit=5)

    for chat in recent_chats:
        # Highlight current chat
        icon = None
        if chat["id"] == st.session_state["chat"].id:
            icon = ":material/edit:"
        st.button(
            chat["title"],
            key=f"chat_{chat['id']}",
            on_click=load_chat,
            args=[chat["id"]],
            use_container_width=True,
            icon=icon,
        )

chat_col, col2 = st.columns([3, 1])

with col2:
    with st.expander("Upload a file"):
        uploaded_file = st.file_uploader("Upload a plain text, markdown or code file")
        st.button("Add to Chat", on_click=handle_add_doc_to_chat)
    st.markdown("""
        Slash commands:

        ```
        /include /path/to/file
        ```   

        Include a file's content into the chat.

        ```
        /include /path *.glob
        ```

        Include several files from path using pattern glob.
    """)

with chat_col:
    # Display chat messages from history on app rerun
    for message in st.session_state["chat"].messages:
        if message["role"] == "system":
            with st.expander("View system prompt"):
                st.markdown(message["content"])
        elif message["role"] == "file":
            with st.chat_message(message["role"], avatar=":material/upload_file:"):
                st.markdown(f"Included `{message['name']}` in chat.")
                with st.expander("View file content"):
                    st.markdown(f"```{message['ext']}\n{message['data']}\n```")
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Generate a reply and add to history
    if st.session_state["generate_assistant"]:
        st.session_state["generate_assistant"] = False

        # Using the .empty() container ensures that once the
        # model starts returning content, we replace the spinner
        # with the streamed content. We then also need to write
        # out the full message at the end (for some reason
        # the message otherwise disappears).
        with st.chat_message("assistant").empty():
            with st.spinner("Thinking...", show_time=False):
                message = st.write_stream(stream_model_response())
            st.session_state["chat"].messages.append(
                {"role": "assistant", "content": message}
            )
            st.write(message)
            save_current_chat()
            st.rerun()

    # Allow user to regenerate the last response.
    if st.session_state["chat"].messages[-1]["role"] == "assistant":
        left, right = st.columns([3, 1])
        with left:
            used_tokens_holder = st.empty()
            used_tokens_holder.caption(
                f"Used tokens: {st.session_state['used_tokens']}"
            )
        with right:
            st.button(
                "Regenerate",
                key="regenerate",
                on_click=regenerate_last_response,
                type="tertiary",
                icon=":material/refresh:",
            )

st.chat_input("Enter prompt here...", key="user_prompt", on_submit=handle_submit_prompt)

# Update the used tokens with the latest value after
# generating a new response.
try:
    used_tokens_holder.caption(f"Used tokens: {st.session_state['used_tokens']}")
except NameError:
    pass
