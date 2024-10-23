from django.http import HttpResponse
import pymongo
from django.shortcuts import render
from django.template import loader
from messages_tagger.models import *
from django.contrib.auth.models import User
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from django.contrib.auth import login as auth_login ## para que la función login() no moleste con el nombre de la funcion
from django.contrib.auth import logout as auth_logout #igual
import datetime
from django.db.models import Max, Q
import pytz
import html
import re
from functools import reduce

N = 1000
sent_limit = 200

def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def logout(request):
    auth_logout(request)
    return redirect('login_index')

def login_index(request):
    print(f"IP {get_client_ip(request)} - login")
    return render(request, 'login.html')

def login(request):
    username = request.POST.get('username', "")
    password = request.POST.get('password', "")
    print(username)
    print(password)
    user = authenticate(username=username, password=password)
    if user is not None and user.is_active:
        # Correct password, and the user is marked "active"
        auth_login(request, user)
        # Redirect to a success page.
        return redirect('index')

    return redirect('login_index')

@login_required
def new_evaluation(request):
    if request.method == 'POST':
        conv_id = int(request.POST.get('submit_button_click'))
        user = request.user

        if not Evaluation.objects.filter(Q(mumuki_id=conv_id) & Q(annotator=user)):
            eval = Evaluation(
                mumuki_id=conv_id,
                annotator=user,
                pub_date=datetime.datetime.now(),
                semantic_score=int(request.POST.get('semantic')),
                content_score=int(request.POST.get('content')),
                completeness_score=int(request.POST.get('completeness')),
                summary_score=int(request.POST.get('summary')),
                comparison_score=int(request.POST.get('comparison'))
            )
            eval.save()

        group = LastTagged.objects.get(user=user).group
        curr_id = LastTagged.objects.filter(group=group).aggregate(Max('next_id'))['next_id__max']
        LastTagged.objects.filter(user=user).update(next_id=curr_id+1)

    return redirect('index')

@login_required
def index(request):
    print(f"IP {get_client_ip(request)} - index")

    def md_translate(text):
        for fragment in re.findall(r'\*\*\S[^\*]*\*\*', text):
            text = text.replace(fragment, '<b>'+fragment[2:-2]+'</b>')
            # text = text.replace(fragment, fragment[2:-2])
        for fragment in re.findall(r'\*\S[^\*]*\*', text):
            text = text.replace(fragment, '<em>'+fragment[1:-1]+'</em>')
            # text = text.replace(fragment, fragment[1:-1])
        return text
    
    def truncate(text):
        return text if len(text) <= sent_limit else text[:sent_limit]+'...'
    
    def locate_sent(text, sent, color):
        tags = ['<br>', '<em>', '</em>', '<b>', '</b>']
        tags_idx = reduce(lambda idxs, tag: idxs+[(match.start(), match.end()) for match in re.finditer(tag, text)], tags, [])
        tags_idx.sort(key=lambda loc: loc[0])

        stripped = text
        for tag in tags:
            stripped = stripped.replace(tag, '')
        if (sent in stripped):
            ok_idx = (stripped.index(sent), stripped.index(sent) + len(sent))
            for tag_idx in tags_idx:
                tag_len = tag_idx[1] - tag_idx[0]
                if tag_idx[0] < ok_idx[0]:
                    ok_idx = (ok_idx[0] + tag_len, ok_idx[1] + tag_len)
                elif ok_idx[0] < tag_idx[0] < ok_idx[1]:
                    ok_idx = (ok_idx[0], ok_idx[1] + tag_len)
                elif ok_idx[1] < tag_idx[0]:
                    exit
            
            text = text[:ok_idx[0]] + f'<span style="background-color: {color};">' + text[ok_idx[0]:ok_idx[1]] + '</span>' + text[ok_idx[1]:]
        
        return text

    user = request.user 
    last_id_object = LastTagged.objects.filter(user=user)[0]
    if(last_id_object.next_id == N+1):
        return HttpResponse("No hay más conversaciones por analizar")
    
    nextConv = Conversation.objects.get(order_id=last_id_object.next_id)
    full_text = nextConv.full_text
    sent_ok = html.escape(nextConv.ok_sent1)
    sent_rej = html.escape(nextConv.rej_sent1)

    context = {
        "id": nextConv.order_id,
        "mumuki_id": nextConv.mumuki_id,
        "user_done": round(nextConv.order_id*100/N, 1),
        "messages": [locate_sent(locate_sent(md_translate(html.escape(msg).replace("\n","<br>\n")), sent_ok[:sent_limit], '#25e75e'), sent_rej[:sent_limit], '#f95c5c') for msg in full_text.split("\n\n")],
        "sent_ok": truncate(sent_ok),
        "sent_rej": truncate(sent_rej)
    }

    template = loader.get_template('index.html')
    return HttpResponse(template.render(context, request))

def tutorial(request):
    print(f"IP {get_client_ip(request)} - tutorial")
    return render(request, 'tutorial.html')