from django import forms
from models import PhotoUploads

class UploadPictureForm(forms.ModelForm):
    """Form model for uploading photos."""

    class Meta:
        model = PhotoUploads
