from django.shortcuts import render
from django.shortcuts import redirect
import time
from .models import Eval
# from src.summarize import summarize

def demo(request):
    if(request.method == 'GET'):
        summary = ""
        document = ""
    else:
        document = request.POST['document']
        summary = "summary"
        # summary = summarize.summarize(document)['pgn']

    return render(request, 'demo.html', {'summary': summary, 'document': document})
def eval(request):
    
    eval_label = request.POST['eval']
    if eval_label == '최악':
        score = 0
    elif eval_label == '미흡':
        score = 1
    elif eval_label == '보통' :
        score = 2
    elif eval_label == '좋음' :
        score = 3
    elif eval_label == '최고' :
        score = 4
    post = Eval()
    post.score = score
    post.name = request.POST['name']
    post.sum = request.POST['sum']
    post.doc = request.POST['doc']
    post.save()
    return redirect('demo')