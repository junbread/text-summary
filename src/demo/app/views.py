from django.shortcuts import render, redirect
from django.http import HttpResponse

from .models import Eval

from src.summarize.summarize import Summarizer
from src.dataset.util import ExamplePicker

import time
import json
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent

summarizer = Summarizer()
article_picker = ExamplePicker(data_path=project_root.joinpath('data', 'original').as_posix())


def demo(request):
    return_object = {
        'article': None,
        'summary': None,
        'options': ["baseline", "textrank"]
    }

    if request.method == 'GET':
        article = ""  # article selection module

        return_object['article'] = article_picker.pick_random_article()

    if request.method == 'POST':
        article = request.POST.get('article', '')
        options = request.POST.getlist('option', [])

        if len(article):
            summary = summarizer.summarize(article, options)

        return_object['article'] = article
        return_object['options'] = options
        return_object['summary'] = summary

    return render(request, 'demo.html', return_object)


def eval(request):
    if request.method == 'POST':
        record = Eval()
        print(request.POST.keys())
        score = {key.replace('score-',''): request.POST[key] for key in request.POST.keys() if key.startswith('score')}
        print(score)
        record.score = json.dumps(score)
        record.name = request.POST['name']
        record.sum = request.POST['sum']
        record.doc = request.POST['doc']
        record.save()

    return redirect('demo')

def avg_eval(request):
    records = Eval.objects.all()
    scores = [json.loads(r.score) for r in records]

    score_pgn = sum([int(s['baseline']) for s in scores]) / len(scores)
    score_textrank = sum([int(s['textrank']) for s in scores]) / len(scores)

    return HttpResponse("score_pgn: {} score_textrank: {}".format(score_pgn, score_textrank))