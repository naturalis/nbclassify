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
from rest_framework import generics, permissions, viewsets, renderers, status
from rest_framework.response import Response
from rest_framework.decorators import detail_route

from orchid.forms import UploadPictureForm
from orchid.models import Photo, Identity
from orchid.serializers import PhotoSerializer, IdentitySerializer

CONFIG_FILE = os.path.join(settings.BASE_DIR, 'orchid', 'config.yml')
TAXA_DB = os.path.join(settings.BASE_DIR, 'orchid', 'taxa.db')
ANN_DIR = os.path.join(settings.BASE_DIR, 'orchid', 'orchid.ann')

# Threshold for significant MSE values.
MSE_LOW = 0.0001


class PhotoViewSet(viewsets.ModelViewSet):
    """View and edit photos."""
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
            roi = roi.split(',')
            roi = [int(x) for x in roi]

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

class IdentityViewSet(viewsets.ModelViewSet):
    """Identify photos and view photo identities."""
    queryset = Identity.objects.all()
    serializer_class = IdentitySerializer
    permission_classes = (permissions.IsAuthenticatedOrReadOnly,)

    @detail_route(methods=['get'], renderer_classes=(renderers.JSONRenderer,
        renderers.TemplateHTMLRenderer))
    def info(self, request, *args, **kwargs):
        identity = self.get_object()
        info = eol_orchid_species_info(str(identity))
        return Response(info, template_name="orchid/eol_species_info.html")

def home(request):
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

            return HttpResponseRedirect(reverse('orchid:identify', args=(photo.id,)))
        else:
            data['error_message'] = "Please select a valid image file."

    data['form'] = UploadPictureForm()
    return render(request, "orchid/home.html", data)

def identify(request, photo_id):
    """Photo upload confirmation page.

    Display the photo with ID `photo_id`, let the user select the region of
    interest, if any, and display a button with which to start the
    identification process.
    """
    if not session_owns_photo(request, photo_id):
        raise Http404

    photo = get_object_or_404(Photo, pk=photo_id)
    data = {}
    data.update(csrf(request))

    data['photo'] = photo
    data['roi'] = None

    # Set the ROI if one was already set.
    if photo.roi:
        roi = photo.roi.split(",")
        x,y,w,h = [int(x) for x in roi]
        data['roi'] = {'x':x, 'y':y, 'w':w, 'h':h, 'x2':x+w, 'y2':y+h}

    return render(request, "orchid/identify.html", data)

def photo_identity(request, photo_id):
    """Return the identification result for a photo."""
    # Only allow viewing of own photos.
    if not session_owns_photo(request, photo_id):
        raise Http404

    photo = get_object_or_404(Photo, pk=photo_id)
    ids = photo.identities.all()
    data = {'identities': ids, 'mse_low': MSE_LOW}
    return render(request, "orchid/identities.html", data)

def classify_image(classifier, image_path, ann_dir):
    """Classify an image using a classfication hierarchy,

    Arguments are an instance of ImageClassifier `classifier`, file path to
    the image file `image_path`, and the path to the directory containing the
    artificial neural networks `ann_dir` for the specified classfication
    hierarchy set in `classifier`.

    Returns the classfications as a list of dictionaries, where each dictionary
    maps each rank to the corresponding taxon. An additional key ``error``
    specifies the mean square error for the entire classfication. The
    classifications returned are ordered by mean square error.
    """
    classes, errors = classifier.classify_with_hierarchy(image_path, ann_dir)

    # Check for failed classification.
    if not classes[0]:
        return []

    # Calculate the mean square error for each classification path.
    errors_classes = [(sum(e)/len(e),c) for e,c in zip(errors, classes)]

    # Get the level names.
    ranks = classifier.get_classification_hierarchy_levels()

    # Create a list of all classifications.
    classes = []
    for error, classes_ in sorted(errors_classes):
        class_dict = {'error': error}
        for rank, taxon in zip(ranks, classes_):
            class_dict[rank] = taxon
        classes.append(class_dict)

    return classes

def photo(request, photo_id):
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

    return render(request, "orchid/photo.html", data)

def delete_photo(request, photo_id):
    """Delete a photo and its related objects."""
    # Only allow deletion of own photos.
    if not session_owns_photo(request, photo_id):
        raise Http404

    photo = get_object_or_404(Photo, pk=photo_id)

    # Delete the photo. Because of the models.photo_delete_hook(), the
    # actual image file will also be deleted.
    photo.delete()

    # Return the result.
    return HttpResponse(json.dumps({'stat': 'success'}),
        content_type="application/json")

def my_photos(request):
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
    return render(request, "orchid/my_photos.html", data)

def session_owns_photo(request, photo_id):
    """Test if the current session own photo with ID `photo_id`.

    Returns True if the session owns the photo, False otherwise.
    """
    try:
        return int(photo_id) in request.session['photos']
    except:
        return False

def get_session_photo_ids(request):
    """Return the photo IDs for the current session."""
    try:
        return list(request.session['photos'])
    except:
        return []

def json_get_session_data(request):
    """Return data for the current user session.

    Data contains the photo IDs for the current user. Data is returned in JSON
    format.
    """
    data = {}
    data['photos'] = get_session_photo_ids(request)
    return HttpResponse(json.dumps(data), content_type="application/json")

def javascript(request):
    """Return Django parsed JavaScript."""
    data = {}
    return render(request, "orchid/javascript.js", data,
        content_type="application/javascript")

def query_eol(query, options, taxon_concept=None, exact=False):
    """Return species info from EOL.org.

    This generator searches EOL with `query` and returns each result. Search
    results can be filtered by taxon concept ID `taxon_concept`.
    """
    params = {
        'q': query,
        'exact': 'true' if exact else 'false'
    }
    if taxon_concept:
        params['filter_by_taxon_concept_id'] = taxon_concept

    # Get the first species ID.
    try:
        url = "http://eol.org/api/search/1.0.json?{0}".\
            format(urllib.urlencode(params))
        rsp = json.load(urllib2.urlopen(url))
    except:
        return

    for result in rsp['results']:
        url = "http://eol.org/api/pages/1.0/{0}.json?{1}".\
            format(result['id'], urllib.urlencode(options))
        yield json.load(urllib2.urlopen(url))

def eol_orchid_species_info(query):
    """Return species info from EOL.org.

    Searches for `query` and returns the first result as a dictionary.
    """
    iucn_status = re.compile(r'\(([A-Z]{2})\)')

    options = {
        'images': 8,
        'videos': 0,
        'sounds': 0,
        'maps': 0,
        'text': 3,
        'iucn': 'true',
        'subjects': 'TaxonBiology|Description|Distribution',
        'details': 'true',
        'vetted': 2,
        'cache_ttl': 300
    }

    # We're only interested in orchids.
    taxon_concept = 8156

    # Get only the first result.
    eol_results = query_eol(query, options, taxon_concept)
    try:
        data = eol_results.next()
    except StopIteration:
        return None

    # Set some extra values.
    scientificName = data['scientificName'].split()
    data['canonicalName'] = ' '.join(scientificName[:2])
    data['describedBy'] = ' '.join(scientificName[2:])
    data['imageObjects'] = []
    data['textObjects'] = []
    data['iucn'] = None
    for obj in data['dataObjects']:
        try:
            if obj['title'] == "IUCNConservationStatus":
                data['iucn'] = obj
                data['iucn']['danger_status'] = iucn_status.\
                    search(obj['description']).\
                    group(1) in ('VU','EN','CR','EW','EX')
                continue
        except:
            pass

        if "StillImage" in obj['dataType']:
            data['imageObjects'].append(obj)

        elif "Text" in obj['dataType']:
            # Skip non-English texts for now.
            if 'language' in obj and obj['language'] != 'en':
                continue

            data['textObjects'].append(obj)

    return data
