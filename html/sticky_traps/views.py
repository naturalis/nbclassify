# coding=utf-8
import json
import os
# import re
# import urllib
# import urllib2
import math

from django.http import HttpResponse, HttpResponseRedirect, Http404, HttpResponseServerError
from django.core.urlresolvers import reverse
from django.shortcuts import render, get_object_or_404
from django.views.decorators import csrf
from django.conf import settings
from django.forms import modelformset_factory
# from nbclassify.classify import ImageClassifier
# from nbclassify.db import session_scope
# from nbclassify.functions import open_config
# from rest_framework import generics, permissions, mixins, viewsets, renderers, status
# from rest_framework.response import Response
# from rest_framework.decorators import detail_route

from forms import ImageForm, VeldData
from models import Photo, Veld
from sticky_traps import analyse_photo
from sticky_traps import find_insects

OUTPUT_VELD = os.path.join(settings.BASE_DIR, 'sticky_traps', 'results', 'veld_data.txt')
OUTPUT_FOTO = os.path.join(settings.BASE_DIR, 'sticky_traps', 'results', 'foto_data.txt')
OUTPUT_OPMERKINGEN = os.path.join(settings.BASE_DIR, 'sticky_traps', 'results', 'opmerkingen.txt')

testfile = os.path.join(settings.BASE_DIR, 'sticky_traps', 'tf.txt')
tf = open(testfile, "a")

import logging
logging.basicConfig(filename=os.path.join(settings.BASE_DIR, 'sticky_traps','pylog.log'), level=logging.DEBUG) #DELETE LOGFILE SETTINGS
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
            datum = veld_form.cleaned_data['Plaatsings_datum']
            if Veld.objects.filter(Veld_identificatie_code=veldnr, Plaatsings_datum=datum).exists():
                data['error_message'] = "De informatie voor dit veld op deze datum is al eerder ingevuld."
            else:
                veld_ingevuld = veld_form.save()
                field_id = veld_ingevuld.id
                for form in foto_form:

                    foto = form.save()
                    foto.veldnr = veldnr
                    foto.unieke_veld_code = field_id
                    foto.datum = datum
                    foto.save()


                return HttpResponseRedirect(reverse('sticky_traps:results', args=(field_id,)))
        else:
            data['error_message'] = "Het formulier is nog niet goed ingevuld."
    else:
        veld_form = VeldData(prefix = 'veld1')
        foto_form = FotoFormSet(queryset=Photo.objects.none(), prefix= "foto's")

    data['veld_form'] = veld_form
    data['foto_form'] = foto_form
    return render(request, "sticky_traps/upload.html", data)


def results(request, field_id):
    data = {}
    velden = Veld.objects.all().values()
    veld_nummers = Veld.objects.all().values("id")
    opgeslagen = list(Veld.objects.filter(id=field_id).values('Opgeslagen'))[0]

    #TODO zorg ervoor dat deze code weer werkt zoals in eerste instantie de bedoeling was

    # Uploads that contain images that cannot be analysed, because of failure to find all corners, are raised as an error_message
    # and the user is recommended to remake the picture of the trap. All of the uploaded images (including the faulty one) are deleted from the database
    # to refrain from getting the "De informatie voor dit veld op deze datum is al eerder ingevuld."-error_message. The images can then be uploaded again with the same information.
    if opgeslagen.get('Opgeslagen')==False:
        gemiddeld_oppervlak_over_veld, variance, valnrfout, output = generate_output(field_id)
        if valnrfout == "":
            data["oppervlak"] = gemiddeld_oppervlak_over_veld
            data["variance"] = variance
            op = "".join(output)
            tb = op.replace("'", "").replace(", ", "<br>")
            data["output"] = tb
        else:
            data['error_message'] = "De foto met het valnummer "+valnrfout+" kan niet worden geanalyseerd doordat\
             de hoeken van de val niet te vinden zijn. Dit kan komen door reflectie van licht op de val, of doordat de val niet helemaal op de foto staat.\
             Probeer het opnieuw door een andere foto te maken en deze te uploaden, samen met de andere geüploade foto's."
            data['oppervlak'] = "N/A"
            data['variance'] = "N/A"

    else:
        if valnrfout == "":
            gemiddeld_oppervlak_over_veld, variance, valnrfout, output = generate_output(field_id)
            veld_object = Veld.objects.get(id=field_id)
            data["oppervlak"] = veld_object.gemiddeld_oppervlak_over_veld
            data["variance"] = veld_object.variance
            data["output"] = output
        else:
            data['error_message'] = "De foto met het valnummer "+valnrfout+" kan niet worden geanalyseerd doordat\
             de hoeken van de val niet te vinden zijn. Dit kan komen door reflectie van licht op de val, of doordat de val niet helemaal op de foto staat.\
              Probeer het opnieuw door een andere foto te maken en deze te uploaden, samen met de andere geüploade foto's."
            data['oppervlak'] = "N/A"
            data['variance'] = "N/A"


    return render(request, "sticky_traps/results.html", data)


# ---------
# Functions
# ---------

# If not all corners of an image can be found it removes the saved data and stops the rest of the code from being run. The trapnumber is returned to inform the user of the faulty image.
def generate_output(field_id):
    fotos_voor_veld = list(Photo.objects.filter(unieke_veld_code=field_id).values())
    Foto_output_file = open(OUTPUT_FOTO, "a+")
    avg_area_list = []
    output = []
    for item in fotos_voor_veld:
        if len(item.get('foto')) != 0:
            itempath=(os.path.join(settings.BASE_DIR, "sticky_traps/uploads/", item.get('foto')))
            insect_informatie, nocorners = analyse_photo(itempath)
            if nocorners == False:
                valnrfout = ""
                Foto_output_file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n"%(item.get('Val_nummer'), item.get('veldnr'), item.get('datum'),
                insect_informatie["average_area"], insect_informatie["number_of_insects"], insect_informatie["smaller_than_4"],
                insect_informatie["between_4_and_10"], insect_informatie["larger_than_10"]))
                avg_area_list.append(insect_informatie["average_area"])
                output.append("\t"+str(item.get('Val_nummer')) +"\t\t"+ str(insect_informatie["number_of_insects"]) +"\t\t"+ str(insect_informatie["smaller_than_4"]) +"\t\t"+
                str(insect_informatie["between_4_and_10"]) +"\t\t"+ str(insect_informatie["larger_than_10"]) +"\n")
            else:
                valnrfout = str(item.get('Val_nummer'))
                print(valnrfout)
                gemiddeld_oppervlak = None
                variance = None
                insect_informatie = None
                Photo.objects.filter(unieke_veld_code=field_id).delete()
                Veld.objects.filter(id=field_id).delete()
    Foto_output_file.close()
    try:
        veld_output = list(Veld.objects.filter(id=field_id).values())[0]
        veld_object = Veld.objects.get(id=field_id)
    except:
        veld_output = None
        veld_object = None
    try:
        gemiddeld_oppervlak = sum(avg_area_list) / float(len(avg_area_list))
        ssd = sum([(x- gemiddeld_oppervlak )**2 for x in avg_area_list])
        variance = math.sqrt(ssd / (len(avg_area_list) - 1))
    except:
        gemiddeld_oppervlak= None
        variance= None
    try:
        veld_object.Opgeslagen=True
        veld_object.gemiddeld_oppervlak_over_veld=gemiddeld_oppervlak
        veld_object.variance=variance
        veld_object.save()
        Veld_output_file = open(OUTPUT_VELD, "a+")
        Veld_output_file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" %(veld_output.get('Veld_identificatie_code'), veld_output.get('Locatie'),
        veld_output.get('Plaatsings_datum'), veld_output.get('Beheer_type'), veld.output.get('Bemesting'), veld_output.get('Beweiding'),veld_output.get('Maaien'),
        veld_output.get('Minimale_hoogte_gras'), veld_output.get('Maximale_hoogte_gras'),veld_output.get('Hoeveelheid_biodiversiteit'), gemiddeld_oppervlak, variance))
        Veld_output_file.close()
        Opmerkingen_output_file = open(OUTPUT_OPMERKINGEN, "a+")
        Opmerkingen_output_file.write("%s\t%s\n"%(veld_output.get('Veld_identificatie_code'), veld_output.get('Opmerkingen_en_bijzonderheden')))
        Opmerkingen_output_file.close()
    except:
        pass
    return gemiddeld_oppervlak, variance, valnrfout, output