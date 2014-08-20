from django.db import models

class Photos(models.Model):
    """Model for uploaded photos."""
    photo = models.ImageField(upload_to='uploads')
