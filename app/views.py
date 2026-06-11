from django.shortcuts import render,HttpResponse
from src.rag_service import RAGService
import os 

# Create your views here.
def index(request):

    if request.method == "POST":
        query = request.POST.get("query")
        # rag = RAGService()
        # rag.index_directory("../books")
        print(os.listdir("../books"))
        print(query)
        print()

    return render(request,"index.html")

def greet(request,name):
    return HttpResponse(f"Hello {name}")