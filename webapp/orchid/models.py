# import the required modules
from django.db import models
from time import time
import random

# function to create a path for the uploaded file
def get_upload_file_name(instance, filename):
    return "uploaded_files/%s" % (filename)

# The Orchid model
class Orchid(models.Model):
    ''' The title field (For some reson the form only works with more than 1 field)
    So give it a title was the best option. This title will never be used'''
    title = models.CharField(max_length=200)
    
    # The picture field. This field will be used for uploading a picture
    picture = models.FileField(upload_to=get_upload_file_name)