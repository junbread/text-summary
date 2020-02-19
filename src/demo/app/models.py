from django.conf import settings
from django.db import models
from django.utils import timezone


class Eval(models.Model):
    score = models.TextField()
    name = models.CharField(max_length=200)
    sum = models.TextField()
    doc = models.TextField()
    created_date = models.DateTimeField(
            default=timezone.now)
    published_date = models.DateTimeField(
            blank=True, null=True)

    def publish(self):
        self.published_date = timezone.now()
        self.save()

    def __str__(self):
        return self.name