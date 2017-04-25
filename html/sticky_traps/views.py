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


# -----------------------------
# View sets for the OrchID API
# -----------------------------

def generate_output(field_id):
    print(str(field_id))
    fotos_voor_veld = list(Photo.objects.filter(veldnr=field_id))
    alle_fotos = Photo.objects.all().values()
    print(alle_fotos)
    print(fotos_voor_veld)

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
            veld_ingevuld = veld_form.save()
            field_id = veld_ingevuld.id
            for form in foto_form:
                form.veld_id=field_id
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
    generate_output(field_id)
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
