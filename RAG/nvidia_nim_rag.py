import streamlit as st
import os
from pathlib import Path
import tempfile
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader, PyMuPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')

llm = ChatNVIDIA(model="nvidia/neva-22b")




def _save_uploaded_pdf(uploaded_file) -> str:
	"""Save uploaded PDF to a temp directory and return the file path."""
	upload_dir = Path(".tmp_uploads")
	upload_dir.mkdir(parents=True, exist_ok=True)
	file_path = upload_dir / uploaded_file.name
	with open(file_path, "wb") as f:
		f.write(uploaded_file.getbuffer())
	return str(file_path)


def _repair_pdf_with_pikepdf(src_path: str) -> str | None:
	"""Attempt to repair a PDF using pikepdf. Returns repaired path or None."""
	try:
		import pikepdf
	except Exception:
		return None
	try:
		repaired_path = str(Path(src_path).with_suffix(".repaired.pdf"))
		with pikepdf.open(src_path) as pdf:
			pdf.save(repaired_path)
		return repaired_path
	except Exception:
		return None


def _load_pdf_best_effort(pdf_path: str):
	"""Try multiple loaders, repair, and OCR fallback to extract text."""
	# 1) Try fast loaders first
	loaders = [
		("PyPDFLoader", lambda p: PyPDFLoader(p).load()),
		("PyMuPDFLoader", lambda p: PyMuPDFLoader(p).load()),
	]
	for name, loader in loaders:
		try:
			docs = loader(pdf_path)
			if docs and sum(len(d.page_content) for d in docs) > 100:
				return docs, name
		except Exception:
			pass

	# 2) Try repair with pikepdf then reload
	repaired = _repair_pdf_with_pikepdf(pdf_path)
	if repaired:
		for name, loader in loaders:
			try:
				docs = loader(repaired)
				if docs and sum(len(d.page_content) for d in docs) > 100:
					return docs, f"{name} (repaired)"
			except Exception:
				pass

	# 3) OCR fallback (Unstructured with hi_res)
	try:
		docs = UnstructuredPDFLoader(pdf_path, strategy="hi_res", ocr=True, ocr_language="eng").load()
		if docs and sum(len(d.page_content) for d in docs) > 100:
			return docs, "UnstructuredPDFLoader (OCR)"
	except Exception:
		pass

	return [], "failed"

# testing vector embedding function perfectly working to determine that is corrupted pdf files
# def vector_embedding():
#     if "vectors" not in st.session_state:
#         st.session_state.embeddings = NVIDIAEmbeddings()
        
#         # Load PDF
#         st.session_state.loader = PyPDFLoader("data/demo.pdf")
#         st.session_state.docs = st.session_state.loader.load()
        
#         if not st.session_state.docs:
#             st.error("No pages loaded from 'data/demo.pdf'")
#             return
            
#         # Debug info
#         st.info(f"Loaded {len(st.session_state.docs)} pages")
#         st.write("First page preview:")
#         st.write(st.session_state.docs[0].page_content[:500] + "...")
#         st.write("**Content length:**", len(st.session_state.docs[0].page_content))
        
#         # Text splitting with fallback
#         try:
#             st.session_state.text_splitter = RecursiveCharacterTextSplitter(
#                 chunk_size=1000,
#                 chunk_overlap=200,
#                 separators=["\n\n", "\n", ". ", " ", ""]
#             )
#             st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
            
#             if not st.session_state.final_documents:
#                 st.warning("Recursive splitter failed. Trying simple splitter...")
                
#                 st.session_state.simple_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#                 st.session_state.final_documents = st.session_state.simple_splitter.split_documents(st.session_state.docs)
                
#             st.info(f"Created {len(st.session_state.final_documents)} text chunks")
            
#         except Exception as e:
#             st.error(f"Text splitting error: {str(e)}")
#             return
            
#         if not st.session_state.final_documents:
#             st.error("No text chunks produced from the document.")
#             return
            
#         # Create vector store
#         st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
#         st.success("Vector store created successfully!")


# main repair vector embedding function to fix the issue for user
def vector_embedding(pdf_path: str | None = None):
	if "vectors" not in st.session_state:
		# Use a supported embeddings model explicitly
		st.session_state.embeddings = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")

		# Decide which PDF to use
		selected_path = pdf_path or "data/demo-food-repaired.pdf"
		st.info(f"Using PDF: {selected_path}")

		# Load via best-effort pipeline
		docs, how = _load_pdf_best_effort(selected_path)
		if not docs:
			st.error("Failed to extract text from the PDF (even after repair/OCR).")
			return

		st.session_state.docs = docs
		st.info(f"Loaded {len(docs)} pages via {how}")
		st.write("First page preview:")
		st.write(docs[0].page_content[:500] + "...")
		st.write("**Content length (page 1):**", len(docs[0].page_content))

		# Text splitting with fallback
		try:
			st.session_state.text_splitter = RecursiveCharacterTextSplitter(
				chunk_size=1000,
				chunk_overlap=200,
				separators=["\n\n", "\n", ". ", " ", ""]
			)
			st.session_state.final_documents = st.session_state.text_splitter.split_documents(docs)

			if not st.session_state.final_documents:
				st.warning("Recursive splitter failed. Trying simple splitter...")
				st.session_state.simple_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
				st.session_state.final_documents = st.session_state.simple_splitter.split_documents(docs)

			st.info(f"Created {len(st.session_state.final_documents)} text chunks")

		except Exception as e:
			st.error(f"Text splitting error: {str(e)}")
			return

		if not st.session_state.final_documents:
			st.error("No text chunks produced from the document.")
			return

		# Create vector store
		st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
		st.success("Vector store created successfully!")



st.title("Nvidia NIM Demo")

# File upload UI
uploaded = st.file_uploader("Upload a PDF (corrupted or scanned supported)", type=["pdf"])
selected_pdf_path: str | None = None
if uploaded is not None:
	selected_pdf_path = _save_uploaded_pdf(uploaded)
	st.success(f"Uploaded: {Path(selected_pdf_path).name}")

prompt = ChatPromptTemplate.from_template(

    """
    Answer the questions based on the provided context only .
    Please provide the most accurate response based on the question or topic
    Do not include chain-of-thought; answer concisely from the context and cite page numbers.
    <context>
    {context}
    <context>
    Questions: {input}

"""
)

prompt1 = st.text_input("Enter Your questions from documents ")

if st.button("Document Embedding"):
	# main function 
	vector_embedding(selected_pdf_path)
	# testing done hopefully it works into the main function.
	# vector_embedding()
	st.write("FAISS Vector store db is ready using NvidiaEmbedding")

if prompt1:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please create the vector store first by clicking 'Document Embedding'")
    else:
        try:
            document_chain = create_stuff_documents_chain(llm, prompt)  
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever,document_chain)
            with st.spinner("🤔 Processing your question..."):
                start = time.process_time()
                response = retrieval_chain.invoke({'input':prompt1})
                # print(response)
                end_time = time.process_time()-start
                print("Response time: ", end_time)
            
            st.success(f"✅ Response generated in {end_time:.2f} seconds")
            st.write("**Answer:**")
            st.write(response['answer'])


            with st.expander("Document Similarity Search"):
                for i, doc in enumerate(response['context']):
                    st.write(doc.page_content)
                    st.write("--------------------")
        except Exception as e:
            raise e            


