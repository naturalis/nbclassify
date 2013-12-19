# import the required modules
from django.db import models

# function to create a path for the uploaded file
def get_upload_file_name(instance, filename):
    return "uploaded_files/%s" % (filename)

# The Orchid model
class Orchid(models.Model):    
    # The picture field. This field will be used for uploading a picture
    picture = models.FileField(upload_to=get_upload_file_name)