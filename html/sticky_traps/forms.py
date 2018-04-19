from django import forms

from models import Photo
from models import Veld
import datetime



class VeldData(forms.ModelForm):

    class Meta:
        model = Veld
        exclude = ('Opgeslagen','gemiddeld_oppervlak_over_veld', 'variance')

class ImageForm(forms.ModelForm):

    class Meta:
        model = Photo
        fields = ('Val_nummer', 'foto', )
