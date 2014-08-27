from django.conf.urls import patterns, url

from orchid import views

urlpatterns = patterns('',
    url(r'^$', views.home, name='home'),
    url(r'^result/(?P<photo_id>\d+)/$', views.result, name='result'),
    url(r'^identify/(?P<photo_id>\d+)/$', views.identify_ajax, name='identify'),
    url(r'^library/$', views.my_photos, name='library'),
)
