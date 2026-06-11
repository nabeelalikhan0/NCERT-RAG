
import numpy as np
import chromadb
import uuid
from typing import List,Any
import os 


class VectorStore:
    """Manages document embeddings in a ChromaDB vetor store"""

    def __init__(self,collection_name: str = "pdf_documents",persist_directory:str="../data/vector_store"):
        """
        Initialize the vector store

        Args:
            collection_name: Name of the ChromaDB Collection
            persist_directory: Directory to persist the vector store
        """

        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""

        try:
            os.makedirs(self.persist_directory,exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata = {"description":"PDF document embeddings for RAG", "hnsw:space": "cosine"}
            )
            print(f"Vector store initizalized. Collection: {self.collection_name}")
            print(f"Existing documents in collection : {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise 
        
    def add_documents(self,documents:List[Any],embeddings: np.ndarray):
        """
        Add documents add their embeddings to the vector store 

        Args:
            documents: List of langChain documents
            embeddings: Corresponding embeddings for the documents
        """

        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"Adding {len(documents)} documents to vector store...")

        ids = []
        metadatas = []
        documents_text = []
        embeddings_list= [] 

        for i, (doc,embedding) in enumerate(zip(documents,embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)              
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)

            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} docuemtns to vector store")
            print(f"Total documents in colleciton: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

