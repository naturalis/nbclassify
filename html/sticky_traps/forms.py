# Importing useful modules
from django import forms
from models import Photo
from models import Veld


#Creating class for the model Veld.
class VeldData(forms.ModelForm):
    class Meta:
        model = Veld
        exclude = ('Opgeslagen', 'gemiddeld_oppervlak_over_veld', 'variance', 'gem_4_mm', 'gem_4_10_mm', 'gem_10_mm')

# Creating class for the model Photo.
class ImageForm(forms.ModelForm):
    class Meta:
        model = Photo
        fields = ('Val_nummer', 'foto',)
