from django import forms

from sticky_traps.models import Photo

class UploadPictureForm(forms.ModelForm):
    """Form model for uploading photos."""

    class Meta:
        model = Photo
        exclude = ('roi',)
