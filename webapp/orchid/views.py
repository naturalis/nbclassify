from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.template.loader import get_template
from django.shortcuts import render_to_response
from django.template import Context
from django.views.generic.base import TemplateView
from forms import UploadPictureForm
from django.core.context_processors import csrf


# Create your views here.
def welcome(request):
    name = "Mirna"
    age = 21
    return render_to_response('welcome.html')

####### Adjustment Benjamin 5-11-13 #########
def processUpload(request, filename):
    def wrapper():
        with open(filename) as f:
            for line in f:
                pass # doe iets met iedere bestandsregel
    threading.Thread(target = wrapper).start()


def upload(request):
    if request.method == 'POST':
        form = UploadPictureForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            processUpload(request, request.FILES["picture"]) # zie hier nog een extra regel
            return HttpResponseRedirect('/upload_success')
        
    else:
        form = UploadPictureForm()    
    args = {}
    args.update(csrf(request))
    
    args['form'] = UploadPictureForm()
    print args
    return render_to_response('upload.html', args)

def upload_success(request):
    return render_to_response('upload_succes.html')
