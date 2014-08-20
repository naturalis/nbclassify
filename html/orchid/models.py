from django.db import models

class PhotoUploads(models.Model):
    """Model for uploaded photos."""
    photo = models.ImageField(upload_to='uploads')
