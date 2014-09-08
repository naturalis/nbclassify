import json
import os
import re
import urllib
import urllib2

import nbclassify as nbc

from django.http import HttpResponse, HttpResponseRedirect, Http404, HttpResponseServerError
from django.core.urlresolvers import reverse
from django.core.exceptions import PermissionDenied
from django.shortcuts import render, get_object_or_404
from django.core.context_processors import csrf
from django.conf import settings
from orchid.forms import UploadPictureForm
from orchid.models import Photo, Identity
from orchid.classify import ImageClassifier, open_yaml

ORCHID_CONF = os.path.join(settings.BASE_DIR, 'orchid', 'orchid.yml')
TAXA_DB = os.path.join(settings.BASE_DIR, 'orchid', 'taxa.db')
ANN_DIR = os.path.join(settings.BASE_DIR, 'orchid', 'orchid.ann')

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

    If the Identify Photo button is pressed, this view is also loaded in the
    background via an AJAX call. The photo is then classified, each
    classification is saved as an Identity object in the database, and the
    results are returned in HTML format.
    """
    # Only allow identifying of own photos.
    if not session_owns_photo(request, photo_id):
        raise PermissionDenied

    photo = get_object_or_404(Photo, pk=photo_id)
    data = {}
    data.update(csrf(request))

    if request.method == 'POST':
        # The Identify Photo button was pressed. Identify the photo and return
        # the results in HTML format.

        # Delete any existing identities, if any.
        Identity.objects.filter(photo=photo).delete()

        # Get the region of interest if it was set.
        try:
            roi = request.POST['roi']
        except:
            roi = None

        # Set the ROI for the photo.
        photo.roi = roi
        photo.save()

        if roi:
            roi = roi.split(',')
            roi = [int(x) for x in roi]
        else:
            roi = None

        # Classify the photo.
        config = open_yaml(ORCHID_CONF)
        classifier = ImageClassifier(config, TAXA_DB)
        classifier.set_roi(roi)

        try:
            classes = classify_image(classifier, photo.image.path, ANN_DIR)
        except Exception as e:
            return HttpResponseServerError(e)

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

        ids = photo.identity_set.all()
        data = {'identities': ids}
        return render(request, "orchid/result_ajax.html", data)
    else:
        # The Identify Photo button was not pressed. So let the user set the
        # region of interest, if any, and then press that button to start
        # identification.

        data['photo'] = photo
        data['roi'] = None

        # Get the ROI.
        if photo.roi:
            y,y2,x,x2 = photo.roi.split(",")
            data['roi'] = {'y':y, 'y2':y2, 'x':x, 'x2':x2}

        return render(request, "orchid/identify.html", data)

def photo_identity(request, photo_id):
    """Return the identification result for a photo."""
    # Only allow viewing of own photos.
    if not session_owns_photo(request, photo_id):
        raise PermissionDenied

    photo = get_object_or_404(Photo, pk=photo_id)
    ids = photo.identity_set.all()
    data = {'identities': ids}
    return render(request, "orchid/result_ajax.html", data)

def classify_image(classifier, image_path, ann_dir):
    """Classify an image using a classfication hierarchy,

    Arguments are an instance of ImageClassifier `classifier`, file path to
    the image file `image_path`, and the path to the directory containing the
    artificial neural networks `ann_dir` for the specified classfication
    hierarchy set in `classifier`.

    Returns the classfications as a list of dictionaries, where each
    dictionary maps each level to the corresponding classfication. An
    additional key ``error`` specifies the mean square error for the entire
    classfication.
    """
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

def photo(request, photo_id):
    """Display the photo with classification result."""
    # Only allow viewing of own photos.
    if not session_owns_photo(request, photo_id):
        raise PermissionDenied

    photo = get_object_or_404(Photo, pk=photo_id)

    data = {}
    data.update(csrf(request))
    data['photo'] = photo

    # Get the identities from the database.
    data['identities'] = photo.identity_set.all()

    return render(request, "orchid/photo.html", data)

def delete_photo(request, photo_id):
    """Delete a photo and its related objects."""
    # Only allow deletion of own photos.
    if not session_owns_photo(request, photo_id):
        raise PermissionDenied

    data = {}
    data.update(csrf(request))
    photo = get_object_or_404(Photo, pk=photo_id)

    if request.method == 'POST':
        # The user confirmed the deletion.

        # Delete the photo. Because of the models.photo_delete_hook(), the
        # actual image file will also be deleted.
        photo.delete()

        # Return the result.
        return HttpResponse(json.dumps({'stat': 'success'}),
            content_type="application/json")
    else:
        # The user did not confirmed the deletion.

        # Display a confirmation page.
        data['photo'] = photo
        return render(request, "orchid/delete_photo.html", data)

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

def json_get_session_photo_ids(request):
    """Return the photo IDs for the current session in JSON format."""
    return HttpResponse(json.dumps(get_session_photo_ids(request)),
        content_type="application/json")

def query_eol(query, options, taxon_concept=None, exact=False):
    """Return species info from EOL.org.

    Searches EOL with `query` and returns each result in JSON format. Search
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
        raise StopIteration()

    for result in rsp['results']:
        url = "http://eol.org/api/pages/1.0/{0}.json?{1}".\
            format(result['id'], urllib.urlencode(options))
        yield json.load(urllib2.urlopen(url))

def eol_orchid_species_info(request, query):
    """Return species info from EOL.org.

    Searches on species name `species` and returns the first result in HTML
    format. If no results were found, a HTTP 404 not found error is raised.
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

    try:
        eol_results = query_eol(query, options, taxon_concept)
    except Exception:
        raise Http404

    # Get only the first result.
    data = eol_results.next()

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
                data['iucn']['danger_status'] = iucn_status.search(obj['description']).group(1) in ('LC','NT','VU','EN','CR','EW')
                continue
        except:
            pass

        if "StillImage" in obj['dataType']:
            data['imageObjects'].append(obj)

        elif "Text" in obj['dataType']:
            data['textObjects'].append(obj)

    return render(request, "orchid/eol_species_info.html", data)
