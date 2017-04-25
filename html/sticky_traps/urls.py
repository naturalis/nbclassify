from django.conf.urls import patterns, include, url
from rest_framework import routers

from sticky_traps import views

# Create a router and register the viewsets with it.
router = routers.DefaultRouter()


urlpatterns = patterns('',
    url(r'^$', views.home, name='home'),
    url(r'^photo/(?P<photo_id>\d+)/$', views.photo, name='photo'),
    url(r'^library/$', views.my_photos, name='library'),
    url(r'^session_data\.json$', views.json_get_session_data, name='session_data'),
    url(r'^orchid\.js$', views.javascript, name='js'),
    url(r'^upload$', views.upload, name='upload'),
    url(r'^results/(?P<field_id>\d+)/$', views.results, name='results'),

    # The API URLs are determined automatically by the router.
    url(r'^api/', include(router.urls, namespace='api')),
)
