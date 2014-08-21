from django.conf.urls import patterns, include, url

from django.contrib import admin
admin.autodiscover()

urlpatterns = patterns('',
    url(r'^$', 'orchid.views.home'),
    url(r'^orchid/', include('orchid.urls', namespace="orchid")),
    url(r'^admin/', include(admin.site.urls)),
)
