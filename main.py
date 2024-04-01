import core
import streamlit as st
import os
import anthropic
import pyautogui


directory = "./uploads"

def delete_file(file_path):
    try:
        os.remove(file_path)
        update_list(directory)
        st.sidebar.success(f"File '{file_path}' deleted successfully!")
    except Exception as e:
        st.sidebar.error(f"Error deleting file '{file_path}': {e}")
        
def update_list(directory):
    list = os.listdir(directory)
    return list

def restart_ui():
    pyautogui.hotkey("ctrl","F5")

with st.sidebar:

    # Get API KEY
    openai_api_key = st.text_input("OPENAI API Key", key="file_qa_api_key", type="password")

    # Starting the session
    start_session = st.sidebar.button(f"🚀 Start Now", disabled= not openai_api_key)
    
    # Get File List Available)
    files_list = update_list(directory)
    st.sidebar.title("List of Files")
    for file in files_list:
        # Create a delete button for each file
        if st.sidebar.button(f"🗑 { (file[:20-3] + '...') if len(file) > 20 else file }"):
            file_path = os.path.join(directory, file)
            delete_file(file_path)
            restart_ui()
 
st.title("📝 Docuquest")
uploaded_file = st.file_uploader("Upload an document", type=("pdf", "docx","txt"))
question = st.text_input(
    "Ask something about the document",
    placeholder="Can you give me a short summary?",
    disabled=not openai_api_key,
)
start_answer = st.button("💬 Ask")

if uploaded_file:
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"File saved locally at: {file_path}")
    restart_ui()


if uploaded_file and question and not openai_api_key:
    st.info("Please add your OPENAI API key to continue.")

if start_session and start_answer and openai_api_key:
    embedding = core.init_huggingface(openai_api_key)
    document_df = core.input_document()
    chunks_df = core.split_document(document_df)
    db_vector = core.init_chroma(chunks_df, embedding)
    model = core.setup_model(db_vector)
    result = core.start_conversation(model)
    st.write(result["answer"])