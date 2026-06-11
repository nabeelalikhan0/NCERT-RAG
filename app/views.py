from django.shortcuts import render,HttpResponse
from src.rag_service import RAGService
import os 

# Create your views here.
def index(request):
    context = {}
    if request.method == "POST":
        query = request.POST.get("query")
        context["query"] = query
        try:
            rag = RAGService()
            rag.index_directory(r"D:\Test\Django\NCERT Rag\books")
            result = rag.ask(query)
            
            if result:
                context["answer"] = result[0]
                context["metadata"] = result[1]
                context["similarity"] = result[2]
            else:
                context["error"] = "No relevant context found to answer the question."
        except Exception as e:
            print(f"Error processing RAG query: {e}")
            context["error"] = f"An error occurred: {str(e)}"

    return render(request, "index.html", context)

def greet(request,name):
    return HttpResponse(f"Hello {name}")