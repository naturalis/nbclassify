from django import forms

from sticky_traps.models import Photo
from sticky_traps.models import Veld
import datetime



class VeldData(forms.ModelForm):

    class Meta:
        model = Veld
        exclude = ('Opgeslagen','gemiddeld_oppervlak_over_veld', 'variance')

class ImageForm(forms.ModelForm):

    class Meta:
        model = Photo
        fields = ('foto', 'Val_nummer', )
