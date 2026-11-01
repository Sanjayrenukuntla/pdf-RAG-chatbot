
# ============================================================
# 1. IMPORTS
# ============================================================

# os helps us access environment variables, such as our API key.
import os

# BytesIO lets us read uploaded PDF files directly from memory.
from io import BytesIO

# Path helps us work with file and folder locations.
from pathlib import Path

# ChromaDB is our vector database.
import chromadb

# Streamlit creates the web application interface.
import streamlit as st

# load_dotenv reads secret values from our .env file.
from dotenv import load_dotenv

# We use the OpenAI Python library to communicate with Gemini
# through Google's OpenAI-compatible API.
from openai import (
    OpenAI,
    OpenAIError,
    APIStatusError,
    APIConnectionError,
    APITimeoutError
)

# PdfReader extracts text from PDF documents.
from pypdf import PdfReader

# These exceptions help us handle invalid or unreadable PDFs.
from pypdf.errors import PdfReadError

# SentenceTransformer converts text into numerical embeddings.
from sentence_transformers import SentenceTransformer


# ============================================================
# 2. PROJECT CONFIGURATION
# ============================================================

# Get the folder in which this app.py file is located.
# This makes our file paths work even if we launch Streamlit
# from a different working directory.
PROJECT_DIR = Path(__file__).parent

# Name of the collection where our PDF chunks will be stored.
COLLECTION_NAME = "pdf_documents"

# This is the local embedding model we will use.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Gemini model used to generate answers.
GEMINI_MODEL_NAME = "gemini-3.5-flash-lite"

# Maximum number of characters in each text chunk.
CHUNK_SIZE = 1000

# Number of characters shared between consecutive chunks.
# This overlap helps preserve context between chunks.
CHUNK_OVERLAP = 150

# Number of relevant chunks retrieved for each question.
RESULT_COUNT = 3

# Google's OpenAI-compatible API endpoint.
GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)


# ============================================================
# 3. LOAD ENVIRONMENT VARIABLES
# ============================================================

# Load the variables from the .env file in our project folder.
# We never print or display the API key.
load_dotenv(dotenv_path=PROJECT_DIR / ".env")


# ============================================================
# 4. LOAD THE EMBEDDING MODEL
# ============================================================

@st.cache_resource
def get_embedding_model():
    """
    Load the Sentence Transformer model.

    @st.cache_resource tells Streamlit to reuse the loaded model
    instead of downloading/loading it every time the app reruns.

    The model is loaded only when we first need embeddings.
    """

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return model


# ============================================================
# 5. CONNECT TO CHROMADB
# ============================================================

@st.cache_resource
def get_chroma_client():
    """
    Create a persistent ChromaDB client.

    PersistentClient saves our vector database on the computer.
    This means the stored PDF data can survive app restarts.
    """

    database_path = PROJECT_DIR / "chroma_db"

    client = chromadb.PersistentClient(
        path=str(database_path)
    )

    return client


# ============================================================
# 6. SPLIT PDF TEXT INTO CHUNKS
# ============================================================

def split_text(text):
    """
    Divide a large text document into smaller overlapping chunks.

    Why do we need chunks?

    Sending an entire PDF to an AI model is inefficient.
    Instead, we divide it into smaller sections.

    The overlap helps prevent important context from being
    lost between two consecutive chunks.
    """

    chunks = []

    # Starting character position.
    start = 0

    # Continue until we reach the end of the document.
    while start < len(text):

        # Calculate where the current chunk should end.
        end = min(start + CHUNK_SIZE, len(text))

        # Extract the text between start and end.
        chunk = text[start:end].strip()

        # Only save chunks that contain actual text.
        if chunk:
            chunks.append(chunk)

        # Stop if we have reached the end of the document.
        if end == len(text):
            break

        # Move forward while keeping the required overlap.
        start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# 7. GET THE EXISTING CHROMADB COLLECTION
# ============================================================

def get_saved_collection():
    """
    Check whether a previously processed PDF exists in ChromaDB.

    If the collection exists, return it.
    Otherwise, return None.
    """

    client = get_chroma_client()

    # Get the names of all existing collections.
    collection_names = {
        collection.name
        for collection in client.list_collections()
    }

    # Check whether our PDF collection exists.
    if COLLECTION_NAME not in collection_names:
        return None

    # Return the existing collection.
    return client.get_collection(
        name=COLLECTION_NAME
    )


# ============================================================
# 8. PROCESS THE UPLOADED PDF
# ============================================================

def process_pdf(uploaded_file):
    """
    Complete PDF processing pipeline:

    1. Read the uploaded PDF.
    2. Extract its text.
    3. Split the text into chunks.
    4. Generate embeddings for those chunks.
    5. Store the chunks and embeddings in ChromaDB.
    """

    # --------------------------------------------------------
    # STEP 1: EXTRACT TEXT FROM THE PDF
    # --------------------------------------------------------

    try:

        # Convert the uploaded file into an in-memory byte stream.
        pdf_data = BytesIO(uploaded_file.getvalue())

        # Create a PDF reader object.
        reader = PdfReader(pdf_data)

        # Extract text from every page.
        # If a page contains no extractable text, use an empty string.
        pages_text = [
            page.extract_text() or ""
            for page in reader.pages
        ]

        # Combine all page text into one document.
        text = "\n".join(pages_text)

    except (PdfReadError, ValueError, OSError) as error:

        st.error(
            "Could not read this PDF. "
            "Please check that it is a valid, unencrypted PDF."
        )

        return

    # Check whether the PDF contains extractable text.
    if not text.strip():

        st.error(
            "This PDF has no extractable text. "
            "It may be scanned. OCR is not supported yet."
        )

        return

    # --------------------------------------------------------
    # STEP 2: CLEAN AND SPLIT THE TEXT
    # --------------------------------------------------------

    # Replace unnecessary spaces and newlines with single spaces.
    cleaned_text = " ".join(text.split())

    # Divide the document into smaller chunks.
    chunks = split_text(cleaned_text)

    # Check whether chunking produced any useful content.
    if not chunks:

        st.error("No text could be split into searchable sections.")

        return

    # --------------------------------------------------------
    # STEP 3: GENERATE EMBEDDINGS
    # --------------------------------------------------------

    try:

        with st.spinner(
            "Loading the embedding model and preparing your PDF..."
        ):

            # Load the embedding model.
            model = get_embedding_model()

            # Convert every chunk into a numerical vector.
            # tolist() converts the result into normal Python lists
            # that can be stored in ChromaDB.
            embeddings = model.encode(chunks).tolist()

    except (OSError, RuntimeError, ValueError) as error:

        st.error(
            f"Could not load the embedding model or process the PDF: {error}"
        )

        return

    # --------------------------------------------------------
    # STEP 4: STORE CHUNKS AND EMBEDDINGS IN CHROMADB
    # --------------------------------------------------------

    try:

        # Connect to our persistent vector database.
        client = get_chroma_client()

        # Get all existing collection names.
        collection_names = {
            collection.name
            for collection in client.list_collections()
        }

        # If a previous PDF collection exists, delete it.
        # This version supports one active PDF at a time.
        if COLLECTION_NAME in collection_names:

            client.delete_collection(
                name=COLLECTION_NAME
            )

        # Create a fresh collection for the uploaded PDF.
        collection = client.create_collection(
            name=COLLECTION_NAME
        )

        # Add the chunks and their corresponding embeddings.
        collection.add(

            # Give every chunk a unique ID.
            ids=[
                f"chunk-{index}"
                for index in range(len(chunks))
            ],

            # Store the original text.
            documents=chunks,

            # Store the numerical embeddings.
            embeddings=embeddings
        )

    except (OSError, chromadb.errors.ChromaError, ValueError) as error:

        st.error(
            f"Could not save this PDF to ChromaDB: {error}"
        )

        return

    # --------------------------------------------------------
    # STEP 5: UPDATE THE STREAMLIT INTERFACE
    # --------------------------------------------------------

    # Clear the previous conversation when a new PDF is processed.
    st.session_state.messages = []

    # Show a success message.
    st.success(
        f"Successfully processed {uploaded_file.name}! "
        f"Saved {len(chunks)} text chunks."
    )


# ============================================================
# 9. CONNECT TO GEMINI AND ANSWER QUESTIONS
# ============================================================

def answer_question(question, collection):
    """
    Answer a user's question using the RAG pipeline.

    1. Load the Gemini API key.
    2. Convert the question into an embedding.
    3. Search ChromaDB for relevant chunks.
    4. Send the retrieved chunks and question to Gemini.
    5. Return Gemini's answer.
    """

    # --------------------------------------------------------
    # STEP 1: GET THE GEMINI API KEY
    # --------------------------------------------------------

    # Read the API key from the .env file.
    api_key = os.getenv("GEMINI_API_KEY")

    # Check whether the API key exists.
    if not api_key:

        st.error(
            "GEMINI_API_KEY is missing. "
            "Add it to your project's .env file."
        )

        return None

    # --------------------------------------------------------
    # STEP 2: CONVERT THE QUESTION INTO AN EMBEDDING
    # --------------------------------------------------------

    try:

        # Load our local embedding model.
        model = get_embedding_model()

        # Convert the user's question into a numerical vector.
        question_embedding = model.encode(
            [question]
        )[0].tolist()

        # ----------------------------------------------------
        # STEP 3: RETRIEVE RELEVANT CHUNKS FROM CHROMADB
        # ----------------------------------------------------

        # Search the vector database for the closest matching
        # chunks to the user's question.
        results = collection.query(

            query_embeddings=[
                question_embedding
            ],

            # Retrieve the three most relevant chunks.
            n_results=RESULT_COUNT
        )

        # Extract the actual text from the search results.
        documents = results.get(
            "documents",
            [[]]
        )[0]

        # Check whether any relevant text was found.
        if not documents:

            st.info(
                "No matching text was found in the uploaded PDF."
            )

            return None

        # Combine the retrieved chunks into one context string.
        context = "\n\n".join(documents)

        # ----------------------------------------------------
        # STEP 4: CREATE THE GEMINI CLIENT
        # ----------------------------------------------------

        # We use the OpenAI Python library as a client,
        # but redirect its requests to Google's Gemini API.
        client = OpenAI(

            # Our Gemini API key.
            api_key=api_key,

            # Google's OpenAI-compatible API endpoint.
            base_url=GEMINI_BASE_URL,

            # Stop waiting indefinitely if the API doesn't respond.
            timeout=30.0,

            # Don't repeatedly retry failed requests.
            max_retries=0 
            
        )
        # ----------------------------------------------------
        # STEP 5: SEND THE QUESTION AND CONTEXT TO GEMINI
        # ----------------------------------------------------

        response = client.chat.completions.create(

            # Specify the Gemini model.
            model=GEMINI_MODEL_NAME,

            # Send instructions and the retrieved PDF context.
            messages=[

                {
                    "role": "system",

                    "content": (
                        "You are a helpful PDF question-answering assistant. "

                        "Answer the user's question using only the "
                        "information in the supplied PDF excerpts. "

                        "Do not make up information or use outside knowledge. "

                        "If the excerpts do not contain the answer, "
                        "clearly say that the answer was not found "
                        "in the uploaded PDF. "

                        "Explain your answers in simple, clear language."
                    )
                },

                {
                    "role": "user",

                    "content": (
                        f"PDF EXCERPTS:\n\n"
                        f"{context}\n\n"
                        f"USER QUESTION:\n{question}"
                    )
                }

            ]
        )

        # ----------------------------------------------------
        # STEP 6: EXTRACT GEMINI'S ANSWER
        # ----------------------------------------------------

        answer = response.choices[0].message.content

        # Return the generated answer.
        return answer

    # Handle errors from the Gemini API.
    
    # Handle errors returned by Google's Gemini API.
    except APIStatusError as error:

        st.error(
            f"Gemini API Error: HTTP {error.status_code}"
        )

        # Display Google's actual error message.
        st.code(str(error.message))

        return None

    # Handle API timeout errors.
    except APITimeoutError:

        st.error(
            "Gemini took too long to respond. "
            "Please try again."
        )

        return None

    # Handle internet or connection errors.
    except APIConnectionError as error:

        st.error(
            f"Connection error: {error}"
        )

        return None

    # Handle other OpenAI SDK errors.
    except OpenAIError as error:

        st.error(
            f"API error ({type(error).__name__}): {error}"
        )

        return None

    # Handle unexpected errors.
    except Exception as error:

        st.error(
            f"Unexpected error ({type(error).__name__}): {error}"
        )

        return None


# ============================================================
# 10. STREAMLIT USER INTERFACE
# ============================================================

# Display the main application title.
st.title("PDF Question Answering Assistant")

# Display a short description below the title.
st.write(
    "Upload a PDF document and ask questions about its contents. "
    "Gemini will answer using information retrieved from your PDF."
)


# ------------------------------------------------------------
# PDF UPLOAD SECTION
# ------------------------------------------------------------

# Create a file uploader that accepts PDF files only.
uploaded_file = st.file_uploader(

    "Choose a PDF file",

    type=["pdf"]
)


# Create a button to start PDF processing.
# The button remains disabled until a PDF is uploaded.
if st.button(
    "Process PDF",
    disabled=uploaded_file is None
):

    # Process the uploaded PDF.
    process_pdf(uploaded_file)


# ------------------------------------------------------------
# CHECK WHETHER A PDF HAS BEEN PROCESSED
# ------------------------------------------------------------

# Retrieve the existing collection from ChromaDB.
collection = get_saved_collection()

# If no collection exists, show an instruction to the user.
if collection is None:

    st.info(
        "Upload and process a PDF to get started."
    )


# ------------------------------------------------------------
# CHAT HISTORY
# ------------------------------------------------------------

# Streamlit reruns the Python script when users interact
# with the interface.

# session_state allows us to preserve chat messages
# between these reruns.

if "messages" not in st.session_state:

    # Start with an empty conversation.
    st.session_state.messages = []


# Display all previous messages in the conversation.
for message in st.session_state.messages:

    # Create a chat message container for each message.
    with st.chat_message(message["role"]):

        # Display the message content using Markdown.
        st.markdown(message["content"])


# ------------------------------------------------------------
# USER QUESTION INPUT
# ------------------------------------------------------------

# Create a chat input box at the bottom of the application.
question = st.chat_input(
    "Ask a question about your PDF..."
)


# Check whether the user submitted a question.
if question:

    # Save the user's question in the conversation history.
    st.session_state.messages.append(

        {
            "role": "user",
            "content": question
        }

    )

    # Display the user's question.
    with st.chat_message("user"):

        st.markdown(question)

    # Make sure a PDF has been processed before answering.
    if collection is None:

        st.error(
            "Please upload and process a PDF before asking questions."
        )

    else:

        # Create the assistant's chat message container.
        with st.chat_message("assistant"):

            # Display a loading spinner while Gemini generates
            # the answer.
            with st.spinner(
                "Searching your PDF and generating an answer..."
            ):

                # Run the RAG pipeline.
                answer = answer_question(
                    question,
                    collection
                )

            # Display and save the answer if one was generated.
            if answer:

                st.markdown(answer)

                # Save the assistant's answer in chat history.
                st.session_state.messages.append(

                    {
                        "role": "assistant",
                        "content": answer
                    }

                )


# ============================================================
# END OF APPLICATION
# ============================================================
