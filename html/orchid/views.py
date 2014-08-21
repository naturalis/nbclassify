import os
from time import time

from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from django.shortcuts import render_to_response, get_object_or_404, render
from django.core.context_processors import csrf
from django.contrib import auth
from django.contrib.auth.decorators import login_required
from django.core.files.uploadedfile import UploadedFile

import nbclassify as nbc
from orchid.forms import UploadPictureForm
from orchid.models import Photo, Identity
from orchid.classify import ImageClassifier, open_yaml

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
ORCHID_CONF = os.path.join(BASE_DIR, 'orchid', 'orchid.yml')
TAXA_DB = os.path.join(BASE_DIR, 'orchid', 'taxa.db')
ANN_DIR = os.path.join(BASE_DIR, 'orchid', 'orchid.ann')

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

    # Classify the photo.
    config = open_yaml(ORCHID_CONF)
    classifier = ImageClassifier(config, TAXA_DB)
    classes = classify_image(classifier, photo.photo.path, ANN_DIR)

    # Create identities for this photo.
    for c in classes:
        if not c.get('genus'):
            continue

        id_ = Identity(
            photo=photo,
            genus=c.get('genus'),
            section=c.get('section'),
            species=c.get('species'),
            error=c.get('error')
        )

        # Save the identity into the database.
        id_.save()

    return HttpResponseRedirect(reverse('orchid:result', args=(photo.id,)))

def classify_image(classifier, image_path, ann_dir):
    classes, errors = classifier.classify_with_hierarchy(image_path, ann_dir)

    # Check for failed classification.
    if not classes[0]:
        return []

    # Calculate the mean square error for each classification path.
    errors_classes = [(sum(e)/len(e),c) for e,c in zip(errors, classes)]

    # Get the level names.
    levels = classifier.get_classification_hierarchy_levels()

    # Create a list of all classifications.
    classes = []
    for error, classes_ in sorted(errors_classes):
        class_dict = {'error': error}
        for level, class_ in zip(levels, classes_):
            class_dict[level] = class_
        classes.append(class_dict)

    return classes

def result(request, photo_id):
    photo = get_object_or_404(Photo, pk=photo_id)

    data = {}
    data.update(csrf(request))
    data['photo'] = photo

    # Get the identities from the database.
    data['identities'] = photo.identity_set.all()

    return render_to_response("orchid/result.html", data)
