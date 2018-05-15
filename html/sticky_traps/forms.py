from django import forms

from models import Photo
from models import Veld
import datetime


class VeldData(forms.ModelForm):
    class Meta:
        model = Veld
        exclude = ('Opgeslagen', 'gemiddeld_oppervlak_over_veld', 'variance', 'gem_4_mm', 'gem_4_10_mm', 'gem_10_mm')


class ImageForm(forms.ModelForm):
    class Meta:
        model = Photo
        fields = ('Val_nummer', 'foto',)

        # Uploaden meerdere foto's en het verkrijgen van valnaam via naam van afbeelding
        # class Meta:
        #     model = Photo
        #     fields = ('foto',)
        #     exclude = ('Val_nummer',)
        #     widgets = {'foto': forms.FileInput(attrs={'multiple': True})}