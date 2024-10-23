from messages_tagger.models import *
from django.contrib.auth.models import User
import json

def retrieve():
    groups = [g for g in LastTagged.objects.values_list('group', flat=True).distinct()]
    evaluations = {}
    for group in groups:
        users = LastTagged.objects.filter(group=group).values_list('user', flat=True)
        all_evals = Evaluation.objects.filter(annotator__in=users).order_by('pub_date')

        evals = []
        for eval in all_evals:
            # username = User.objects.get(pk=eval.annotator).username
            evals.append({
                'conv_id': eval.mumuki_id,
                # 'annotator': username,
                'semantic': eval.semantic_score,
                'content': eval.content_score,
                'completeness': eval.completeness_score,
                'summary': eval.summary_score,
                'comparison': eval.comparison_score,
            })
        evaluations[group] = evals
    
    with open('data/results.json', 'w', encoding='utf-8') as fw1:
        json.dump(evaluations, fw1)
        print('Anotaciones exportadas a json.')
        
    

