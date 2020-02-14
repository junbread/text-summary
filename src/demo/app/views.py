from django.shortcuts import render, redirect
from .models import Eval

from src.summarize.summarize import Summarizer

import time

summarizer = Summarizer()

def demo(request):
    return_object = {
        'article': None,
        'summary': None,
        'options': []
    }
    
    if(request.method == 'GET'):
        article = "" # article selection module

    if(request.method == 'POST'):
        article = request.POST['article']
        
        if article is not None:
            summary = summarizer.summarize(article)

    return render(request, 'demo.html', return_object)


def eval(request):

    eval_record = Eval()
    eval_record.score = int(request.POST['score'])
    eval_record.name = request.POST['name']
    eval_record.sum = request.POST['sum']
    eval_record.doc = request.POST['doc']
    eval_record.save()

    return redirect('demo')
