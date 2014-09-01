from django.conf.urls import patterns, url

from orchid import views

urlpatterns = patterns('',
    url(r'^$', views.home, name='home'),
    url(r'^photo/(?P<photo_id>\d+)/$', views.photo, name='photo'),
    url(r'^photo/(?P<photo_id>\d+)/identify/$', views.identify, name='identify'),
    url(r'^photo/(?P<photo_id>\d+)/identity/$', views.photo_identity, name='identity'),
    url(r'^photo/(?P<photo_id>\d+)/delete/$', views.delete_photo, name='delete_photo'),
    url(r'^library/$', views.my_photos, name='library'),
    url(r'^session_photo_ids\.json$', views.json_get_session_photo_ids),
    url(r'^eol_species_info/(?P<query>[\w\s\+]+)/$', views.eol_orchid_species_info, name='eol_species_info'),
)
