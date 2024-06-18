import os
import faiss
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
import azure.cognitiveservices.speech as speechsdk
import speech_recognition as sr

d = 1536
faiss_index = faiss.IndexFlatL2(d)
PERSIST_DIR = "./storage"

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

def saveUploadedFiles(pdf_docs):
    UPLOAD_DIR = 'uploaded_files'
    try:
        for pdf in pdf_docs:
            file_path = os.path.join(UPLOAD_DIR, pdf.name)
            with open(file_path, "wb") as f:
                f.write(pdf.getbuffer())
        return "Done"
    except:
        return "Error"

def doVectorization():    
    try:
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store) 
        documents = SimpleDirectoryReader("./uploaded_files").load_data()
        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context
        )
        index.storage_context.persist()
        return "Done"
    except:
        return "Error"

def fetchData(user_question):
    try:
        vector_store = FaissVectorStore.from_persist_dir("./storage")
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, persist_dir=PERSIST_DIR
        )
        index = load_index_from_storage(storage_context=storage_context)
        query_engine = index.as_query_engine()
        response = query_engine.query(user_question)
        return str(response)
    except:
        return "Error"

def transcribe_audio():
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=20)
    
    st.write("🔄 Transcribing...")
    
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.RequestError:
        return "API unavailable or unresponsive"
    except sr.UnknownValueError:
        return "Unable to recognize speech"

#============================================================================================================
WelcomeMessage = """
Hello, I am your pdf voice chatbot. Please upload your PDF documents and start asking questions to me. 
I would try my best to answer your question from the documents
"""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content=WelcomeMessage)
    ]

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
speech_config.speech_synthesis_voice_name = "en-US-AriaNeural"   
speech_config.speech_synthesis_language = "en-US"

speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

def main():
    load_dotenv()  

    st.set_page_config(
        page_title="Chat with multiple PDFs",
        page_icon=":sparkles:"
    )

    st.header("Chat with single or multiple PDFs :sparkles:")   

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)     

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", 
            accept_multiple_files=True
        )

        if st.button("Process"):
            with st.spinner("Processing"):
                IsFilesSaved = saveUploadedFiles(pdf_docs)
                if IsFilesSaved == "Done":
                    IsVectorized = doVectorization()
                    if IsVectorized == "Done":
                        st.session_state.isPdfProcessed = "done"
                        st.success("Done!")
                    else:
                        st.error("Error! in vectorization")
                else:
                    st.error("Error! in saving the files")

    if st.button("Start Asking Question"):
        st.write("🎤 Recording started...Ask your  question")
        transcription = transcribe_audio()
        st.write("✅ Recording ended")

        st.session_state.chat_history.append(HumanMessage(content=transcription))

        with st.chat_message("Human"):
            st.markdown(transcription)
        
        with st.chat_message("AI"):
            with st.spinner("Fetching data ..."):
                response = fetchData(transcription)
                st.markdown(response)    
                
        result = speech_synthesizer.speak_text_async(response).get()
        st.session_state.chat_history.append(AIMessage(content=response))
        
    if "WelcomeMessage" not in st.session_state:
        st.session_state.WelcomeMessage = WelcomeMessage
        result = speech_synthesizer.speak_text_async(WelcomeMessage).get()

#============================================================================================================
if __name__ == '__main__':
    main()