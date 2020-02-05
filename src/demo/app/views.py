from django.shortcuts import render
from src.summarize import summarize

def demo(request):
    if(request.method == 'GET'):
        summary = ""
        document = ""
        answer = ""
    else:
        document = request.POST['document']
        answer = request.POST['answer']
        summary = summarize.summarize(document)['pgn']

    return render(request, 'demo.html', {'summary': summary, 'document': document, 'answer': answer})
