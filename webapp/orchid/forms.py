from django import forms
from models import Orchid

class UploadPictureForm(forms.ModelForm):
    
    class Meta:
        model = Orchid
        fields = ('title', 'picture')