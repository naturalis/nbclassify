from django.conf.urls import patterns, url

from orchid import views

urlpatterns = patterns('',
    url(r'^$', views.home, name='home'),
    url(r'^photo/(?P<photo_id>\d+)/$', views.photo, name='photo'),
    url(r'^photo/(?P<photo_id>\d+)/identify/$', views.identify, name='identify'),
    url(r'^photo/(?P<photo_id>\d+)/identity/$', views.photo_identity, name='identity'),
    url(r'^photo/(?P<photo_id>\d+)/delete/$', views.delete_photo, name='delete_photo'),
    url(r'^library/$', views.my_photos, name='library'),
)
