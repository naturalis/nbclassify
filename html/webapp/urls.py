from django.conf.urls import include, url, static
from django.conf import settings

from django.contrib import admin
admin.autodiscover()

urlpatterns = patterns('',
    #url(r'^$', 'orchid.views.home'),
    #url(r'^orchid/', include('orchid.urls', namespace="orchid")),
    url(r'^admin/', include(admin.site.urls)),
    url(r'^', include('sticky_traps.urls', namespace="sticky_traps")),
)

# Serving files uploaded by a user during development.
if settings.DEBUG:
    urlpatterns += static.static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
