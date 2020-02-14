from django.shortcuts import render, redirect
from .models import Eval

from src.summarize.summarize import Summarizer

import time
summarizer = Summarizer()

def demo(request):
    opt_transformer_checked = None
    opt_textrank_checked = None
    summary = None
    document = None
    opt_trn = None
    opt_textrank_checked = None
    if(request.method == 'GET'):
        summary = "##요약이 여기에 표시됩니다!"
        document = ""
        opt_transformer_checked = "checked"
        opt_textrank_checked = "checked"
        trns = 0
        rnd = 0
    else:
        trns = False
        rnd = False
        document = request.POST['document']
        answer = request.POST['answer']
        summary = summarizer.summarize(document)['pgn']
        
    return render(request, 'demo.html', {'summary': summary, 'document': document, 'summary_trns': trns, 'summary_rnd':rnd, 'opt_trns_checked':opt_transformer_checked, 'opt_rnd_checked':opt_textrank_checked})
  
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