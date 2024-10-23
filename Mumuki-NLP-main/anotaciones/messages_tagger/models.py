from django.db import models
from django.contrib.auth.models import User

class LastTagged(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, default = 1)
    group = models.IntegerField(default=1)
    next_id = models.IntegerField(default=1)

class Evaluation(models.Model):
    mumuki_id = models.IntegerField(default=1)
    annotator = models.ForeignKey(User, on_delete=models.CASCADE, default = 1)
    pub_date = models.DateTimeField('date published')
    semantic_score = models.IntegerField(default=3)
    content_score = models.IntegerField(default=3)
    completeness_score = models.IntegerField(default=3)
    summary_score = models.IntegerField(default=3)
    comparison_score = models.IntegerField(default=3)

class Conversation(models.Model):
    order_id = models.IntegerField(default=1, unique=True)
    mumuki_id = models.IntegerField()
    full_text = models.TextField()
    kw1 = models.TextField()
    kw2 = models.TextField()
    kw3 = models.TextField()
    kw4 = models.TextField()
    kw5 = models.TextField()
    ok_sent1 = models.TextField()
    ok_sent2 = models.TextField()
    ok_sent3 = models.TextField()
    rej_sent1 = models.TextField()
    rej_sent2 = models.TextField()
    rej_sent3 = models.TextField()