import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os


load_dotenv()

api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not api_key:
    api_key = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

# ---- LLM ----
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)   # configure model
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

# ---- Prompt ----
prompt = PromptTemplate(
    template="Summarize the following document in simple and clear points:\n\n{text}",
    input_variables=["text"]
)

chain = prompt | model | parser
# ---- Streamlit UI ----
st.title("Paper Summarizer Tool")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf", accept_multiple_files=False, help="Only PDF files are supported")


def clean_text(text):
    return text.encode("utf-8", "ignore").decode("utf-8")

if uploaded_file is not None:
    
    # Save file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # ---- Load PDF ----
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    # ---- Split text (important for large PDFs) ----
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100
    )

    docs = text_splitter.split_documents(documents)

    # Combine all chunks into one text (simple approach)
    full_text = " ".join([doc.page_content for doc in docs])

    # if st.button("Summarize"):
    #     with st.spinner("Generating summary..."):
            
    #         chain = prompt | model | parser
    #         result = chain.invoke({"text": full_text})

    #         st.subheader("📌 Summary")
    #         st.write(result)



    # ---- Split into chunks ----
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=50
    )

    chunks = text_splitter.split_text(full_text)

    if st.button("Generate Summary"):

        summaries = []

        with st.spinner("Processing chunks..."):
            for chunk in chunks:
                clean_chunk = clean_text(chunk) 
                summary = chain.invoke({"text": clean_chunk})
                summaries.append(summary)

        # ---- Combine summaries ----
        final_input = " ".join(summaries)

        with st.spinner("Generating final summary..."):
            final_summary = chain.invoke({"text": final_input})

        st.subheader("📌 Final Summary")
        st.write(final_summary)






            #footer
st.markdown(
    """
    <hr>
    <p style='text-align: center; font-size: 15px; color: gray;'>
        Developed by: <b> Pervez Abbas </b> ( AI intern at SDA Technology Hub Gilgit, Gilgit Baltistan Pakistan )
    </p>
    """,
    unsafe_allow_html=True
)