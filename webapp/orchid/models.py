from django.db import models
from time import time
import random

def get_upload_file_name(instance, filename):
    return "uploaded_files/%s" % (filename)

# Create your models here.
class Orchid(models.Model):
    title = models.CharField(max_length=200)
    picture = models.FileField(upload_to=get_upload_file_name)
    
    def __unicode__(self):
        return self.title    