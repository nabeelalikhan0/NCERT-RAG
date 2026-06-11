import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List,Dict,Any,Tuple
from sklearn.metrics.pairwise import cosine_similarity
from .Vector_store import VectorStore
from .embeddings_manager import EmbeddingManager



class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self,vector_store:VectorStore,embedding_manager:EmbeddingManager):
        """
        Initialize the retriever 

        Args:
            vectore_store: Vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """

        self.vector_store = vector_store
        self.embedding_manager = embedding_manager


    def retrieve(self,query:str,top_k:int=5,scorethreshold:float = 0.0,) -> List[Dict[str,Any]]:
        """
        Retrieve relevant document for a query

        Args:
            query: The search query
            top_k: Number of top results to return 
            scorethreshold: Minimum similarity threshold

        Returns:
            List of dictionaries containing retrieved documents and metadata
        """

        print(f"Retrieving document for query: {query}")
        print(f"Top_k : {top_k}, Score Threshold: {scorethreshold}")

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]


        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i,(doc_id,document,metadata,distance) in enumerate(zip(ids,documents,metadatas,distances)):
                    similarity_score = 1-distance

                    if similarity_score >= scorethreshold:
                        retrieved_docs.append({
                            "id":doc_id,
                            "content":document,
                            "metadata":metadata,
                            "similarity":similarity_score,
                            "distance":distance,
                            "rank":i+1
                        })

                print(f"Retrieved {len(retrieved_docs)} document (after filtering)")
                return retrieved_docs
            else:
                print("No document found")
                return retrieved_docs
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
        

