from django import forms

from orchid.models import Photo

class UploadPictureForm(forms.ModelForm):
    """Form model for uploading photos."""

    class Meta:
        model = Photo
        exclude = ('roi',)
