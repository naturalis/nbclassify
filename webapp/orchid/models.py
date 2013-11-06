from django.db import models
from time import time

def get_upload_file_name(instance, filename):
    return "uploaded_files/%s_%s" % (str(time()).replace('.', '_'), filename)

# Create your models here.
class Orchid(models.Model):
    title = models.CharField(max_length=200)
    picture = models.FileField(upload_to=get_upload_file_name)