from django.db import models
from time import time
import random

var_part = random.randint(0, 1000)

def get_upload_file_name(instance, filename):
    return "uploaded_files/%s_%s" % (var_part, filename.replace(' ', '_'))

# Create your models here.
class Orchid(models.Model):
    title = models.CharField(max_length=200)
    picture = models.FileField(upload_to=get_upload_file_name)