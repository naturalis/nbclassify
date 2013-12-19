# Import the required modules
from django import forms
from models import Orchid

# Class for uploading pictures
class UploadPictureForm(forms.ModelForm):
    
    # The meta data
    class Meta:
        # The used model, Orchids in this case
        model = Orchid
        # The field to be displayed
        '''WARNING: the comma at after 'picture' is required to make it work.
        Otherwise django will cause a FieldError'''
        fields = ('picture',)