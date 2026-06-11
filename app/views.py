from django.shortcuts import render,HttpResponse
from src.rag_service import RAGService


# Create your views here.
def index(request):

    if request.method == "POST":
        query = request.POST.get("query")
        print(query)

    return render(request,"index.html")

def greet(request,name):
    return HttpResponse(f"Hello {name}")