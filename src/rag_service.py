# services/rag_service.py

from .Vector_store import VectorStore
from .process_pdfs import process_all_pdfs,split_documents
from .embeddings_manager import EmbeddingManager
from .RagRetriver import RAGRetriever
from .RAG import rag_simple
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY
)

class RAGService:

    def __init__(self):

        self.embedding_manager = EmbeddingManager()

        self.vector_store = VectorStore()

        self.retriever = RAGRetriever(
            self.vector_store,
            self.embedding_manager
        )

    def index_directory(self, directory):

        docs = process_all_pdfs(directory)

        chunks = split_documents(docs)

        embeddings = self.embedding_manager.generate_embeddings(
            [d.page_content for d in chunks]
        )

        self.vector_store.add_documents(
            chunks,
            embeddings
        )

    def ask(self, query):

        return rag_simple(
            query=query,
            retriever=self.retriever,
            llm=llm
        )