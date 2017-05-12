import json
import os
# import re
# import urllib
# import urllib2
import math

from django.http import HttpResponse, HttpResponseRedirect, Http404, HttpResponseServerError
from django.core.urlresolvers import reverse
from django.shortcuts import render, get_object_or_404
from django.core.context_processors import csrf
from django.conf import settings
from django.forms import modelformset_factory
# from nbclassify.classify import ImageClassifier
# from nbclassify.db import session_scope
# from nbclassify.functions import open_config
# from rest_framework import generics, permissions, mixins, viewsets, renderers, status
# from rest_framework.response import Response
# from rest_framework.decorators import detail_route

from sticky_traps.forms import ImageForm, VeldData
from sticky_traps.models import Photo, Veld
from sticky_traps.analyze import analyse_photo


OUTPUT_VELD = os.path.join(settings.BASE_DIR, 'sticky_traps', 'results', 'veld_data.txt')
OUTPUT_FOTO = os.path.join(settings.BASE_DIR, 'sticky_traps', 'results', 'foto_data.txt')


# --------------------------
# Standard Sticky traps views
# --------------------------

def home(request):
    """Display the home page."""

    return render(request, "sticky_traps/home.html")


def upload(request):

    data = {}
    FotoFormSet = modelformset_factory(Photo, form=ImageForm, extra=10)

    if request.method == 'POST':

        veld_form = VeldData(request.POST, prefix = 'veld1')
        foto_form = FotoFormSet(request.POST, request.FILES, queryset=Photo.objects.none(), prefix = "foto's")

        if veld_form.is_valid() and foto_form.is_valid():
            veldnr=veld_form.cleaned_data['Veld_identificatie_code']
            """
            if Veld.objects.filter(Veld_identificatie_code=veldnr).exists():
                data['error_message'] = "De informatie voor dit veld is al eerder ingevuld."
            else:
            """
            #TODO bovenstaande code uit aanhalingstekens halen en onderstaande code weer 1 tab verder zetten
            veld_ingevuld = veld_form.save()
            field_id = veld_ingevuld.id
            for form in foto_form:
                foto = form.save()
                foto.veldnr = field_id
                foto.save()
            return HttpResponseRedirect(reverse('sticky_traps:results', args=(field_id,)))
        else:
            data['error_message'] = "Het formulier is nog niet goed ingevuld."
            # print veld_form.errors
            # print foto_form.errors
    else:
        veld_form = VeldData(prefix = 'veld1')
        foto_form = FotoFormSet(queryset=Photo.objects.none(), prefix= "foto's")

    data['veld_form'] = veld_form
    data['foto_form'] = foto_form
    return render(request, "sticky_traps/upload.html", data)



def results(request, field_id):
    data = {}
    velden = Veld.objects.all().values()
    # print(velden)
    veld_nummers = Veld.objects.all().values("id")
    # print(veld_nummers)
    opgeslagen = list(Veld.objects.filter(id=field_id).values('Opgeslagen'))[0]
    gemiddeld_oppervlak_over_veld, variance = generate_output(field_id)
    data["oppervlak"] = gemiddeld_oppervlak_over_veld
    data["variance"] = variance
    #TODO zorg ervoor dat deze code weer werkt zoals in eerste instantie de bedoeling was
    """
    if opgeslagen.get('Opgeslagen')==False:
        print "We gaan door met het opslaan van de gegevens"
        generate_output(field_id)
    else:
        print "Dit object is al opgeslagen in de outputbestanden."
    """
    return render(request, "sticky_traps/results.html", data)

# ---------
# Functions
# ---------

def generate_output(field_id):

    fotos_voor_veld = list(Photo.objects.filter(veldnr=field_id).values())
    Foto_output_file = open(OUTPUT_FOTO, "a+")
    Veld_output_file = open(OUTPUT_VELD, "a+")
    avg_area_list = []
    for item in fotos_voor_veld:
        itempath=os.path.abspath(os.path.join("media/", item.get('foto')))
        insect_informatie = analyse_photo(itempath)
        # print insect_informatie
        # print insect_informatie.get("geschat_aantal_insecten")
        Foto_output_file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%(item.get('Val_nummer'), item.get('veldnr'),insect_informatie["total_area"],
                    insect_informatie["number_of_insects"], insect_informatie["smaller_than_4"], insect_informatie["between_4_and_10"],
                    insect_informatie["larger_than_10"]))
        avg_area_list.append(insect_informatie["total_area"])
    veld_output = list(Veld.objects.filter(id=field_id).values())[0]
    veld_object = Veld.objects.get(id=field_id)
    gemiddeld_oppervlak_over_veld = sum(avg_area_list) / float(len(avg_area_list))
    ssd = sum([(x- gemiddeld_oppervlak_over_veld )**2 for x in avg_area_list])
    variance = math.sqrt(ssd / (len(avg_area_list) - 1))
    #print veld_object
    veld_object.Opgeslagen=True
    veld_object.save()
    #print(veld_output)
    Veld_output_file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" %(
        veld_output.get('Veld_identificatie_code'), veld_output.get('Locatie'), veld_output.get('Beheer_type'),
        veld_output.get('Plaatsings_datum'),veld_output.get('Beweiding'),
        veld_output.get('Maaien'), veld_output.get('Minimale_hoogte_gras'), veld_output.get('Maximale_hoogte_gras'),
        veld_output.get('Hoeveelheid_biodiversiteit'), gemiddeld_oppervlak_over_veld, variance
        ))
    return gemiddeld_oppervlak_over_veld, variance
