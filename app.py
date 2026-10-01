import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv
import pandas as pd
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# -----------------------------
# Page setup & styling (UI only)
# -----------------------------

st.set_page_config(
    page_title="MR Analyst",
    page_icon="📊",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif&family=DM+Sans:wght@400;500;600&display=swap');

:root {
    --ink: #14213d; --muted: #5b6577; --paper: #f6f4ff; --card: #ffffff; --line: #dde2ea;
    --teal: #0f766e; --amber: #f5a524; --indigo: #4f46e5; --pink: #ec4899;
    --sky: #0ea5e9; --violet: #7c3aed; --mint: #10b981;
}
@keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes drift { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(18px,-14px) scale(1.08); } }
@keyframes shift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(245,165,36,.55); } 50% { box-shadow: 0 0 0 8px rgba(245,165,36,0); } }

html, body, .stApp, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
.stApp {
    background:
        radial-gradient(700px 380px at 100% 0%, rgba(236,72,153,.14), transparent 60%),
        radial-gradient(700px 380px at 0% 25%, rgba(14,165,233,.14), transparent 60%),
        radial-gradient(800px 480px at 60% 100%, rgba(124,58,237,.12), transparent 60%),
        var(--paper);
}
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
.block-container { max-width: 820px; padding-top: 2.5rem; padding-bottom: 5rem; }
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-thumb { background: linear-gradient(var(--indigo), var(--pink)); border-radius: 10px; }

/* Hero */
.mr-hero {
    position: relative; overflow: hidden;
    background: linear-gradient(120deg, #14213d, #3730a3, #7c3aed, #be185d, #3730a3);
    background-size: 300% 300%; animation: shift 14s ease infinite, fadeUp .7s ease both;
    border-radius: 24px; padding: 2.6rem 2.3rem 2.2rem; margin-bottom: 2rem;
    border-bottom: 6px solid var(--amber);
    box-shadow: 0 18px 40px rgba(79,70,229,.28);
}
.mr-orb { position: absolute; border-radius: 50%; filter: blur(8px); opacity: .35; animation: drift 9s ease-in-out infinite; }
.mr-orb.o1 { width: 190px; height: 190px; right: -50px; top: -60px; background: radial-gradient(circle, #f9a8d4, transparent 70%); }
.mr-orb.o2 { width: 150px; height: 150px; right: 120px; bottom: -70px; background: radial-gradient(circle, #7dd3fc, transparent 70%); animation-delay: -4s; }
.mr-badge {
    display: inline-block; position: relative; margin-bottom: 1rem; padding: .3rem .85rem;
    font-size: .8rem; font-weight: 600; letter-spacing: .04em; color: #fff;
    background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.3); border-radius: 999px;
}
.mr-hero h1 {
    position: relative; font-family: 'Instrument Serif', serif; font-weight: 400;
    font-size: 3.6rem; line-height: 1.03; color: #fff; margin: 0 0 .7rem 0; padding: 0;
}
.mr-hero h1 .grad {
    background: linear-gradient(90deg, #fde68a, #fbcfe8, #bae6fd);
    -webkit-background-clip: text; background-clip: text; color: transparent; font-style: italic;
}
.mr-hero p { position: relative; color: #e0e7ff; font-size: 1.05rem; margin: 0; max-width: 34rem; }
.mr-chips { position: relative; display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.3rem; }
.mr-chips span {
    padding: .35rem .8rem; font-size: .85rem; font-weight: 500; color: #fff;
    background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.28);
    border-radius: 999px; backdrop-filter: blur(4px); transition: transform .2s, background .2s;
}
.mr-chips span:hover { transform: translateY(-3px) rotate(-2deg); background: rgba(255,255,255,.28); }

/* Section headings with numbered badges */
.mr-section {
    display: flex; align-items: center; gap: .9rem; margin: 2.6rem 0 .9rem 0;
    padding: .9rem 1.1rem; background: rgba(255,255,255,.7); backdrop-filter: blur(6px);
    border: 1px solid var(--line); border-radius: 16px; animation: fadeUp .6s ease both;
    box-shadow: 0 4px 14px rgba(20,33,61,.05);
}
.mr-num {
    flex: 0 0 auto; width: 46px; height: 46px; border-radius: 14px; display: grid; place-items: center;
    font-size: 1.3rem; color: #fff; background: linear-gradient(135deg, var(--teal), var(--sky));
    box-shadow: 0 6px 14px rgba(15,118,110,.3); transform: rotate(-6deg);
}
.mr-section.s2 .mr-num { background: linear-gradient(135deg, var(--sky), var(--indigo)); }
.mr-section.s3 .mr-num { background: linear-gradient(135deg, var(--indigo), var(--violet)); }
.mr-section.s4 .mr-num { background: linear-gradient(135deg, var(--pink), var(--amber)); }
.mr-section.s5 .mr-num { background: linear-gradient(135deg, var(--mint), var(--teal)); }
.mr-section h2 { font-family: 'Instrument Serif', serif; font-weight: 400; font-size: 1.8rem; color: var(--ink); margin: 0; padding: 0; }
.mr-section p { color: var(--muted); margin: .15rem 0 0 0; font-size: .93rem; }

/* File uploader */
[data-testid="stFileUploader"] section {
    background: linear-gradient(135deg, #ffffff 0%, #eef2ff 55%, #fdf2f8 100%);
    border: 2px dashed #a5b4fc; border-radius: 18px; padding: 1.5rem; transition: transform .2s, border-color .2s, box-shadow .2s;
}
[data-testid="stFileUploader"] section:hover { border-color: var(--pink); transform: translateY(-2px); box-shadow: 0 10px 24px rgba(236,72,153,.15); }
[data-testid="stFileUploader"] label p { font-weight: 600; color: var(--ink); }
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section div,
[data-testid="stFileUploader"] section span,
[data-testid="stFileUploader"] section small { color: var(--muted) !important; }
[data-testid="stFileUploader"] section svg { fill: var(--indigo) !important; color: var(--indigo) !important; }
[data-testid="stFileUploader"] section button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important; border: none !important; border-radius: 10px !important;
}
[data-testid="stFileUploader"] section button,
[data-testid="stFileUploader"] section button * { color: #ffffff !important; }
[data-testid="stFileUploader"] section button:hover { background: linear-gradient(135deg, #3730a3, #be185d) !important; }
[data-testid="stFileUploaderFile"] { background: #fff; border-radius: 10px; margin-top: .3rem; border-left: 4px solid var(--mint); }
[data-testid="stFileUploaderFile"], [data-testid="stFileUploaderFile"] * { color: var(--ink) !important; }
[data-testid="stFileUploaderFile"] svg { fill: var(--indigo) !important; }

/* Buttons */
.stButton > button {
    position: relative; overflow: hidden; background: var(--card); color: var(--indigo);
    border: 1.5px solid #c7d2fe; border-radius: 12px; padding: .6rem 1.3rem; font-weight: 600;
    transition: transform .15s, box-shadow .15s, background .2s, color .2s, border-color .2s;
}
.stButton > button p, .stButton > button div, .stButton > button span { color: inherit !important; }
.stButton > button::after {
    content: ""; position: absolute; top: 0; left: -75%; width: 50%; height: 100%;
    background: linear-gradient(120deg, transparent, rgba(255,255,255,.55), transparent); transform: skewX(-20deg);
}
.stButton > button:hover {
    border-color: transparent; color: #fff; background: linear-gradient(135deg, #4f46e5, #ec4899);
    transform: translateY(-2px); box-shadow: 0 10px 20px rgba(124,58,237,.3);
}
.stButton > button:hover::after { left: 130%; transition: left .6s ease; }
.stButton > button:active { transform: translateY(0) scale(.98); }
.stButton > button:focus-visible { outline: 3px solid var(--amber); outline-offset: 2px; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #0f766e, #0ea5e9); border-color: transparent; color: #fff; animation: pulse 2.6s infinite; }
.stButton > button[kind="primary"]:hover { background: linear-gradient(135deg, #059669, #4f46e5); color: #fff; }

/* Text input */
.stTextInput input {
    background: #fff; border: 1.5px solid #c7d2fe; border-radius: 12px; padding: .75rem 1rem; color: var(--ink);
    transition: box-shadow .2s, border-color .2s;
}
.stTextInput label p { font-weight: 600; }
.stTextInput input:focus { border-color: var(--pink); box-shadow: 0 0 0 4px rgba(236,72,153,.16), 0 8px 20px rgba(236,72,153,.12); }

/* Alerts */
[data-testid="stAlert"], [data-testid="stNotification"] {
    background: #ffffff !important; border: 1px solid var(--line) !important; border-left: 6px solid var(--indigo) !important;
    border-radius: 14px; animation: fadeUp .4s ease both; box-shadow: 0 6px 16px rgba(20,33,61,.10);
}
[data-testid="stAlert"], [data-testid="stAlert"] *,
[data-testid="stNotification"], [data-testid="stNotification"] * { color: var(--ink) !important; font-weight: 500; }
[data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]) { background: #ecfdf5 !important; border-left-color: #10b981 !important; }
[data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]) { background: #fffbeb !important; border-left-color: #f59e0b !important; }
[data-testid="stAlert"]:has([data-testid="stAlertContentError"]) { background: #fef2f2 !important; border-left-color: #ef4444 !important; }
[data-testid="stAlert"]:has([data-testid="stAlertContentInfo"]) { background: #eff6ff !important; border-left-color: #0ea5e9 !important; }

/* Errors / exceptions (tracebacks) */
[data-testid="stException"] {
    background: #fef2f2 !important; border: 1px solid #fecaca !important; border-left: 6px solid #ef4444 !important;
    border-radius: 14px; padding: .6rem 1rem; box-shadow: 0 6px 16px rgba(239,68,68,.12);
}
[data-testid="stException"], [data-testid="stException"] * { color: #7f1d1d !important; }
[data-testid="stException"] code, [data-testid="stException"] pre { background: transparent !important; }

/* Toasts, dialogs and spinners */
[data-testid="stToast"] {
    background: #ffffff !important; border: 1px solid var(--line); border-left: 6px solid var(--indigo);
    border-radius: 14px; box-shadow: 0 12px 30px rgba(20,33,61,.22);
}
[data-testid="stToast"], [data-testid="stToast"] * { color: var(--ink) !important; }
div[role="dialog"] { background: #ffffff !important; border-radius: 18px; }
div[role="dialog"], div[role="dialog"] * { color: var(--ink) !important; }
[data-testid="stSpinner"], [data-testid="stSpinner"] * { color: var(--ink) !important; }

/* Results */
.stApp h3 {
    font-family: 'DM Sans', sans-serif; font-weight: 600; font-size: 1.05rem; color: var(--violet); margin-top: 1rem;
    display: inline-block; padding-bottom: .15rem; border-bottom: 3px solid; border-image: linear-gradient(90deg, var(--violet), var(--pink), transparent) 1;
}
.stApp h2 { font-family: 'Instrument Serif', serif; font-weight: 400; }
pre, code { border-radius: 10px; }
[data-testid="stCode"], .stMarkdown pre { border: 1px solid #c7d2fe; border-left: 5px solid var(--indigo); box-shadow: 0 6px 16px rgba(79,70,229,.08); }
</style>
""", unsafe_allow_html=True)

# Connect to Groq
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)
# Embedding model
@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

embedding_model = load_embedding_model()


# ChromaDB
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="mr_analyst"
)




# -----------------------------
# Page Title
# -----------------------------

st.markdown("""
<div class="mr-hero">
    <div class="mr-orb o1"></div>
    <div class="mr-orb o2"></div>
    <span class="mr-badge">✨ AI-powered analyst workspace</span>
    <h1>MR <span class="grad">Analyst</span></h1>
    <p>Welcome to India's best AI platform for analysts. Upload your files, then ask questions or generate SQL.</p>
    <div class="mr-chips"><span>📄 PDF</span><span>📊 CSV</span><span>🧠 Ask AI</span><span>🗄️ SQL</span></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="mr-section s1">
    <div class="mr-num">📁</div>
    <div>
        <h2>Add your files</h2>
        <p>CSV, Excel, PDF and TXT files are supported for extraction.</p>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# File Upload
# -----------------------------

uploaded_file = st.file_uploader(
    "Kindly Upload your file to begin",
    type=["csv", "xlsx", "xls", "pdf", "docx", "txt"],
    accept_multiple_files=True
)

if st.button("Upload Now"):
    if uploaded_file:
        st.session_state["uploaded_file"] = uploaded_file
        st.success(f"{len(uploaded_file)} file(s) uploaded successfully!")

    else:
        st.warning("Please select at least one file.")

# pdf extraction function
def extract_pdf(file):

    text = ""

    with pdfplumber.open(file) as pdf:

        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text

# CSV extraction function
def extract_csv(file):

    df = pd.read_csv(file)

    return df.to_string(index=False)

# Excel extraction function (.xlsx / .xls) - reads every sheet
def extract_excel(file):

    file.seek(0)

    sheets = pd.read_excel(file, sheet_name=None)

    text = ""

    for sheet_name, df in sheets.items():
        text += f"Sheet: {sheet_name}\n"
        text += df.to_string(index=False) + "\n\n"

    return text

# Plain text extraction function
def extract_txt(file):

    file.seek(0)

    return file.read().decode("utf-8", errors="ignore")

def extract_file(file):

    name = file.name.lower()

    if name.endswith(".csv"):
        return extract_csv(file)

    elif name.endswith(".pdf"):
        return extract_pdf(file)

    elif name.endswith((".xlsx", ".xls")):
        return extract_excel(file)

    elif name.endswith(".txt"):
        return extract_txt(file)

    else:
        raise ValueError("Unsupported file type")

# -----------------------------
# Extract Data
# -----------------------------

st.markdown("""
<div class="mr-section s2">
    <div class="mr-num">⚙️</div>
    <div>
        <h2>Prepare your data</h2>
        <p>Run these in order: extract, create embeddings, then store.</p>
    </div>
</div>
""", unsafe_allow_html=True)

if st.button("Extract Data"):

    if "uploaded_file" not in st.session_state:

        st.warning("Please upload files first.")

    else:

        extracted_data = []

        for file in st.session_state["uploaded_file"]:

            content = extract_file(file)

            extracted_data.append({
                "filename": file.name,
                "content": content
            })

        st.session_state["extracted_data"] = extracted_data

        st.success(
            f"Data extracted from {len(extracted_data)} file(s) successfully!"
        )

def chunk_text(text, chunk_size=500, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def create_embeddings(extracted_data):

    all_chunks = []
    all_embeddings = []

    for document in extracted_data:

        filename = document["filename"]
        content = document["content"]

        # Step 1: Create chunks
        chunks = chunk_text(content)

        # Step 2: Convert chunks into embeddings
        embeddings = embedding_model.encode(chunks)

        # Store chunks and embeddings
        for chunk, embedding in zip(chunks, embeddings):

            all_chunks.append({
                "filename": filename,
                "content": chunk
            })

            all_embeddings.append(embedding)

    return all_chunks, all_embeddings

def store_in_chromadb(chunks, embeddings):

    documents = []
    metadatas = []
    ids = []

    for index, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):

        documents.append(chunk["content"])

        metadatas.append({
            "filename": chunk["filename"],
            "chunk_index": index
        })

        ids.append(
            f"{chunk['filename']}_{index}"
        )

    collection.upsert(
        documents=documents,
        embeddings=[embedding.tolist() for embedding in embeddings],
        metadatas=metadatas,
        ids=ids
    )

    return len(documents)

if st.button("Create Chunks & Embeddings"):

    if "extracted_data" not in st.session_state:

        st.warning("Please extract the data first.")

    else:

        chunks, embeddings = create_embeddings(
            st.session_state["extracted_data"]
        )

        st.session_state["chunks"] = chunks
        st.session_state["embeddings"] = embeddings

        st.success(
            f"Created {len(chunks)} chunks and embeddings successfully!"
        )

# -----------------------------
# Store in ChromaDB
# -----------------------------

if st.button("Store in ChromaDB"):

    if "chunks" not in st.session_state:

        st.warning(
            "Please create chunks and embeddings first."
        )

    else:

        total_chunks = store_in_chromadb(
            st.session_state["chunks"],
            st.session_state["embeddings"]
        )

        st.success(
            f"{total_chunks} chunks stored in ChromaDB!"
        )
# -----------------------------
# Ask AI
# -----------------------------

st.markdown("""
<div class="mr-section s3">
    <div class="mr-num">💬</div>
    <div>
        <h2>Now ask AI about your file</h2>
        <p>Answers come only from the documents you stored.</p>
    </div>
</div>
""", unsafe_allow_html=True)

question = st.text_input(
    "Ask your question here"
)



# -----------------------------
# Ask AI Button
# -----------------------------

def retrieve_top_chunks(question, top_k=3):

    # Convert question into embedding
    question_embedding = embedding_model.encode(
        question
    ).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    return results

# -----------------------------
# Test Retrieval
# -----------------------------

if st.button("Retrieve Top 3 Chunks"):

    if collection.count() == 0:

        st.warning(
            "ChromaDB is empty. Please store your chunks first."
        )

    elif question.strip() == "":

        st.warning(
            "Please enter a question first."
        )

    else:

        results = retrieve_top_chunks(question, top_k=3)

        st.subheader("Top 3 Retrieved Chunks")

        for i, chunk in enumerate(
            results["documents"][0]
        ):

            st.write(f"### Chunk {i + 1}")

            st.write(chunk)

            st.write(
                "Source:",
                results["metadatas"][0][i]["filename"]
            )

# -----------------------------
# Ask AI Button
# -----------------------------

if st.button("Ask AI", type="primary"):

    if question.strip() == "":

        st.warning("Please enter a question first.")

    elif collection.count() == 0:

        st.warning(
            "Please upload, extract and store your files first."
        )

    else:

        # -----------------------------
        # Step 1: Retrieve top 3 chunks
        # -----------------------------

        results = retrieve_top_chunks(
            question,
            top_k=3
        )

        retrieved_chunks = results["documents"][0]

        # -----------------------------
        # Step 2: Create context
        # -----------------------------

        context = "\n\n".join(
            retrieved_chunks
        )

        # -----------------------------
        # Step 3: Show retrieved data
        # -----------------------------

        st.subheader("Retrieved Context")

        for i, chunk in enumerate(retrieved_chunks):

            st.write(f"### Chunk {i + 1}")
            st.write(chunk)

        # -----------------------------
        # Step 4: Call Groq
        # -----------------------------

        response = client.chat.completions.create(

            model="openai/gpt-oss-20b",

            messages=[

                {
                    "role": "system",
                    "content": """
                    You are MR Analyst, an AI assistant for data analysts.

                    Answer the user's question using the provided context.

                    If the answer cannot be found in the context,
                    clearly say that the information is not available
                    in the uploaded documents.

                    Do not make up information.
                    """
                },

                {
                    "role": "user",
                    "content": f"""
                    Context from uploaded documents:

                    {context}

                    User Question:

                    {question}
                    """
                }
            ],

            temperature=0.2
        )

        # -----------------------------
        # Step 5: Display answer
        # -----------------------------

        answer = response.choices[0].message.content

        st.subheader("AI Response")

        st.markdown(answer)


# -----------------------------
# SQL Generator
# -----------------------------

def generate_sql_from_csv(file, question):

    # Reset file pointer
    file.seek(0)

    # Read CSV
    df = pd.read_csv(file)

    # Get column information
    schema = df.dtypes.astype(str).to_dict()

    # Get sample rows
    sample_data = df.head(5).to_dict(orient="records")

    # Create schema text
    schema_text = "\n".join(
        [
            f"{column}: {dtype}"
            for column, dtype in schema.items()
        ]
    )

    # Call Groq
    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[

            {
                "role": "system",
                "content": """
You are MR Analyst, an expert SQL developer.

Your task is to generate SQL queries for MySQL 8.0
based strictly on the provided CSV dataset schema
and sample data.

Rules:

1. Generate ONLY MySQL-compatible SQL.
2. Use only columns that actually exist in the dataset.
3. Do not invent column names.
4. Infer the appropriate table name from the CSV filename.
5. Use correct MySQL syntax.
6. Make the SQL directly runnable in MySQL Workbench.
7. Do not use PostgreSQL, SQL Server, SQLite, or other SQL dialects.
8. If the user's request cannot be answered using the available
   columns, clearly explain what information is missing.
9. Return the SQL inside a single ```sql code block.
10. After the SQL, provide a short explanation of what the query does.
11. Do not generate Python or pandas code.
12. Do not assume columns that are not present in the dataset.

The dataset schema and sample data will be provided by the user.
"""
            },

            {
                "role": "user",
                "content": f"""
CSV Dataset Schema:

{schema_text}

Sample Data:

{sample_data}

User's SQL Requirement:

{question}
"""
            }

        ],

        temperature=0.1
    )

    return response.choices[0].message.content

# -----------------------------
# SQL Generator
# -----------------------------

st.markdown("""
<div class="mr-section s5">
    <div class="mr-num">🗄️</div>
    <div>
        <h2>SQL generator</h2>
        <p>Upload one CSV, describe what you need in the question box, then generate MySQL code.</p>
    </div>
</div>
""", unsafe_allow_html=True)

if st.button("Generate SQL Code", type="primary"):

    # Check if files were uploaded
    if "uploaded_file" not in st.session_state:

        st.warning(
            "Please upload a CSV file first."
        )

    else:

        # Find CSV files
        csv_files = [
            file
            for file in st.session_state["uploaded_file"]
            if file.name.lower().endswith(".csv")
        ]

        # No CSV found
        if len(csv_files) == 0:

            st.warning(
                "SQL Generator works only with CSV files."
            )

        # More than one CSV
        elif len(csv_files) > 1:

            st.warning(
                "Please upload only one CSV file "
                "when using the SQL Generator."
            )

        # Exactly one CSV
        else:

            if question.strip() == "":

                st.warning(
                    "Please enter your SQL requirement "
                    "in the question box first."
                )

            else:

                csv_file = csv_files[0]

                sql_code = generate_sql_from_csv(
                    csv_file,
                    question
                )

                st.subheader("Generated MySQL Code")

                st.markdown(sql_code)

