from dotenv import load_dotenv
import os 
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY
)


def rag_simple(query,retriever,llm,top_k=3):
    ## retriever the context
    results=retriever.retrieve(query, top_k=top_k)
    context="\n\n".join([doc['content'] for doc in results]) if results else ""
    if not context:
        return "No relevant context found to answer the question."

    ## generate the answwer using GROQ LLM
    prompt=f"""Use the following context to answer the question concisely.
    Context:
    {context}

    Question: {query}

    Answer:"""

    response=llm.invoke([prompt.format(context=context,query=query)])
    print(results)
    return response.content,results[0]["metadata"],results[0]["similarity"]
    