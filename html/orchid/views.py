import os
from time import time

from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.shortcuts import render_to_response, get_object_or_404, render
from django.core.context_processors import csrf
from django.contrib import auth
from django.contrib.auth.decorators import login_required
from django.core.files.uploadedfile import UploadedFile

from orchid.forms import UploadPictureForm
from orchid.models import Photo

def home(request):
    data = {}
    data.update(csrf(request))

    if request.method == 'POST':
        form = UploadPictureForm(request.POST, request.FILES)

        if form.is_valid():
            photo = form.save()
            data['photo'] = photo
            return render_to_response("orchid/upload_succes.html", data)
        else:
            data['error_message'] = "Please select a valid image file."

    data['form'] = UploadPictureForm()
    return render_to_response("orchid/home.html", data)

def classify(request, photo_id):
    photo = get_object_or_404(Photo, pk=photo_id)

    # TODO: Classify the photo.

    return HttpResponseRedirect(reverse('orchid:result', args=(photo.id,)))

def result(request, photo_id):
    photo = get_object_or_404(Photo, pk=photo_id)

    data = {}
    data.update(csrf(request))
    data['photo'] = photo

    # TODO: Get the classification from the database.
    data['classification'] = [
        {'genus':'Paphiopedilum', 'section':'Cochlopetalum', 'species':'liemianum', 'error':0.0001},
        {'genus':'Paphiopedilum', 'section':'Cochlopetalum', 'species':'primulinum', 'error':0.0002},
    ]

    return render_to_response("orchid/result.html", data)
