import json
import os
import re
import urllib
import urllib2

from django.http import HttpResponse, HttpResponseRedirect, Http404, HttpResponseServerError
from django.core.urlresolvers import reverse
from django.shortcuts import render, get_object_or_404
from django.core.context_processors import csrf
from django.conf import settings
from django.forms import modelformset_factory
from nbclassify.classify import ImageClassifier
from nbclassify.db import session_scope
from nbclassify.functions import open_config
from rest_framework import generics, permissions, mixins, viewsets, renderers, status
from rest_framework.response import Response
from rest_framework.decorators import detail_route

from sticky_traps.forms import ImageForm, VeldData
from sticky_traps.models import Photo, Veld
from sticky_traps.serializers import PhotoSerializer
from sticky_traps.analyze import analyse_photo


OUTPUT_VELD = os.path.join(settings.BASE_DIR, 'sticky_traps', 'results', 'veld_data.txt')
OUTPUT_FOTO = os.path.join(settings.BASE_DIR, 'sticky_traps', 'results', 'foto_data.txt')

# -----------------------------
# View sets for the OrchID API
# -----------------------------

def generate_output(field_id):

    fotos_voor_veld = list(Photo.objects.filter(veldnr=field_id).values())
    Foto_output_file = open(OUTPUT_FOTO, "a+")
    Veld_output_file = open(OUTPUT_VELD, "a+")
    for item in fotos_voor_veld:
        itempath=os.path.abspath(os.path.join("media/", item.get('foto')))
        analyse_photo(itempath)
        Foto_output_file.write("%s\t%s\t\n"%(item.get('code'), item.get('veldnr')))
    veld_output = list(Veld.objects.filter(id=field_id).values())[0]
    veld_object = Veld.objects.get(id=field_id)
    print veld_object
    veld_object.Opgeslagen=True
    veld_object.save()
    #print(veld_output)
    Veld_output_file.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t\t\t\n" %(
        veld_output.get('Veld_nummer'), veld_output.get('Breedtegraad'), veld_output.get('Lengtegraad'), veld_output.get('Beheer_type'),
        veld_output.get('Plaatsings_datum'), veld_output.get('Verwijderings_datum'), veld_output.get('Locatie_binnen_veld'),
        veld_output.get('Beweiding'), veld_output.get('Maaien'), veld_output.get('Minimale_hoogte_gras'), veld_output.get('Maximale_hoogte_gras'),
        veld_output.get('Hoeveelheid_biodiversiteit')
        ))


# --------------------------
# Standard OrchID views
# --------------------------

def home(request): #let's keep this one entirely.
    """Display the home page."""
    data = {}
    data.update(csrf(request))

    if request.method == 'POST':
        form = UploadPictureForm(request.POST, request.FILES)

        if form.is_valid():
            photo = form.save()

            # Keep track of which photo belongs to which session.
            try:
                request.session['photos'] += [photo.id]
            except:
                request.session['photos'] = [photo.id]

            return HttpResponseRedirect(reverse('sticky_traps:library'))
        else:
            data['error_message'] = "Please select a valid image file."

    data['form'] = UploadPictureForm()
    return render(request, "sticky_traps/home.html", data)


def upload(request):

    data = {}
    FotoFormSet = modelformset_factory(Photo, form=ImageForm, extra=10)

    if request.method == 'POST':

        veld_form = VeldData(request.POST, prefix = 'veld1')
        foto_form = FotoFormSet(request.POST, request.FILES, queryset=Photo.objects.none(), prefix = "foto's")

        if veld_form.is_valid() and foto_form.is_valid():
            veldnr=veld_form.cleaned_data['Veld_nummer']
            """
            if Veld.objects.filter(Veld_nummer=veldnr).exists():
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
            print veld_form.errors
            print foto_form.errors
    else:
        veld_form = VeldData(prefix = 'veld1')
        foto_form = FotoFormSet(queryset=Photo.objects.none(), prefix= "foto's")

    data['veld_form'] = veld_form
    data['foto_form'] = foto_form
    return render(request, "sticky_traps/upload.html", data)



def results(request, field_id):
    velden = Veld.objects.all().values()
    # print(velden)
    veld_nummers = Veld.objects.all().values("id")
    # print(veld_nummers)
    opgeslagen = list(Veld.objects.filter(id=field_id).values('Opgeslagen'))[0]
    generate_output(field_id)
    #TODO zorg ervoor dat deze code weer werkt zoals in eerste instantie de bedoeling was
    """
    if opgeslagen.get('Opgeslagen')==False:
        print "We gaan door met het opslaan van de gegevens"
        generate_output(field_id)
    else:
        print "Dit object is al opgeslagen in de outputbestanden."
    """
    return render(request, "sticky_traps/results.html")



def photo(request, photo_id): #is a lot like the previous function, could get rid of either this one or the previous one for now.
    """Display the photo with classification result."""
    # Only allow viewing of own photos.
    if not session_owns_photo(request, photo_id):
        raise Http404

    photo = get_object_or_404(Photo, pk=photo_id)

    data = {}
    data.update(csrf(request))
    data['photo'] = photo

    # Get the identities from the database.
    data['identities'] = photo.identities.all()

    return render(request, "sticky_traps/photo.html", data)

def photo_identities(request, photo_id):
    """Display the identities for a photo."""
    if not session_owns_photo(request, photo_id):
        raise Http404
    photo = get_object_or_404(Photo, pk=photo_id)
    data = {'identities': photo.identities.all()}
    return render(request, "sticky_traps/identities.html", data)


def my_photos(request): #probably very useful in a modified state.
    """Display the photos that were identified in a session."""
    data = {}
    photos = []
    pks = get_session_photo_ids(request)

    # Get the photos that belong to this session.
    for photo_id in sorted(pks, reverse=True):
        try:
            photo = Photo.objects.get(pk=photo_id)
            photos.append(photo)
        except (KeyError, Photo.DoesNotExist):
            # We can't modify session values directly.
            ids = request.session['photos']
            ids.remove(photo_id)
            request.session['photos'] = ids

    data['photos'] = photos
    return render(request, "sticky_traps/my_photos.html", data)

def javascript(request): #have absolutely no idea what this does....
    """Return Django parsed JavaScript."""
    data = {}
    return render(request, "sticky_traps/orchid.js", data,
        content_type="application/javascript")

def json_get_session_data(request): #useful, will keep
    """Return data for the current user session.

    Data contains the photo IDs for the current user. Data is returned in JSON
    format.
    """
    data = {}
    data['photos'] = get_session_photo_ids(request)
    return HttpResponse(json.dumps(data), content_type="application/json")

# --------------------------
# Functions
# --------------------------

def session_owns_photo(request, photo_id):
    """Test if the current session own photo with ID `photo_id`.

    Returns True if the session owns the photo, False otherwise.
    """
    try:
        return int(photo_id) in request.session['photos']
    except:
        return False #useful probably

def get_session_photo_ids(request):
    """Return the photo IDs for the current session."""
    try:
        return list(request.session['photos'])
    except:
        return []

# TODO: Write the code that calculates and displays the results.
