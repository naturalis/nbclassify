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
from nbclassify.classify import ImageClassifier
from nbclassify.db import session_scope
from nbclassify.functions import open_config
from rest_framework import generics, permissions, mixins, viewsets, renderers, status
from rest_framework.response import Response
from rest_framework.decorators import detail_route

from sticky_traps.forms import UploadPictureForm
from sticky_traps.models import Photo, Identity
from sticky_traps.serializers import PhotoSerializer, IdentitySerializer


# -----------------------------
# View sets for the OrchID API
# -----------------------------

class PhotoViewSet(viewsets.ModelViewSet): #potentially useful for viewing the images uploaded, can be cleaned during second pass.
    """View, edit, and identify photos."""
    queryset = Photo.objects.all()
    serializer_class = PhotoSerializer
    permission_classes = (permissions.AllowAny,)

    @detail_route(methods=['get','post'])
    def identify(self, request, *args, **kwargs):
        """Identify a photo."""
        photo = self.get_object()

        # If the ROI is set, use that. If no ROI is set, then use the existing
        # ROI if any. If the ROI is set, but evaluates to False, then set the
        # ROI to None.
        roi = request.data.get('roi', photo.roi)
        if not roi:
            roi = None

        if photo.roi != roi:
            photo.roi = roi
            photo.save()

        # Set the ROI for the classifier.
        if roi:
            try:
                roi = roi.split(',')
                roi = [int(x) for x in roi]
                assert len(roi) == 4
            except:
                return Response({'roi': "Must be of the format `x,y,width,height`"},
                    status=status.HTTP_400_BAD_REQUEST)

        # Delete all photo identities, if any.
        Identity.objects.filter(photo=photo).delete()

        # Classify the photo.
        config = open_config(CONFIG_FILE)
        classifier = ImageClassifier(config)
        classifier.set_roi(roi)
        classes = classify_image(classifier, photo.image.path, ANN_DIR)

        # Identify this photo.
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

        return self.retrieve(request, *args, **kwargs)

    @detail_route(methods=['get'],
        renderer_classes=(renderers.JSONRenderer,
            renderers.BrowsableAPIRenderer))
    def identities(self, request, *args, **kwargs):
        """List all identifications made for a photo."""
        photo = self.get_object()
        ids = photo.identities.all()
        serializer = IdentitySerializer(ids, many=True,
            context={'request': request})

        data = {'identities': serializer.data}
        return Response(data)

class IdentityViewSet(mixins.RetrieveModelMixin,
                      mixins.DestroyModelMixin,
                      mixins.ListModelMixin,
                      viewsets.GenericViewSet): #not sure what it does specificially, keep it for now.
    """List and view photo identities."""
    queryset = Identity.objects.all()
    serializer_class = IdentitySerializer
    permission_classes = (permissions.AllowAny,)

    @detail_route(methods=['get'],
        renderer_classes=(renderers.JSONRenderer,
        renderers.BrowsableAPIRenderer))
    def eol(self, request, *args, **kwargs):
        """Get taxon information from EOL.org."""
        identity = self.get_object()
        info = eol_orchid_species_info(str(identity))
        return Response(info)

# --------------------------
# Standard OrchID views
# --------------------------

def home(request): #let's keep this one enirely.
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
