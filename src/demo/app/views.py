from django.shortcuts import render
from django.shortcuts import redirect
import time
from .models import Eval
# from src.summarize import summarize

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
        summary = "summarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummarysummary"
        try:
            if "trans" in request.POST['sum_opt']:
                opt_transformer_checked = "checked"
                trns = True
            if "rnd" in request.POST['sum_opt']:
                opt_textrank_checked = "checked"
                rnd = True
        except:
            pass
        # summary = summarize.summarize(document)['pgn']

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