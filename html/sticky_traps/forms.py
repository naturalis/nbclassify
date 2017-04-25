from django import forms

from sticky_traps.models import Photo
from sticky_traps.models import Veld



class VeldData(forms.ModelForm):

    class Meta:
        model = Veld
        exclude = ()

class ImageForm(forms.ModelForm):
    
    class Meta:
        model = Photo
        fields = ('foto', 'code' )
