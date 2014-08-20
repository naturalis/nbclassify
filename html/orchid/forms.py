from django import forms
from models import Photos

class UploadPictureForm(forms.ModelForm):
    """Form model for uploading photos."""

    class Meta:
        model = Photos
