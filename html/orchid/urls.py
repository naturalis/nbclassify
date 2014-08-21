from django.conf.urls import patterns, url

from orchid import views

urlpatterns = patterns('',
    url(r'^$', views.home, name='home'),
    url(r'^result/(?P<photo_id>\d+)/$', views.result, name='result'),
    url(r'^classify/(?P<photo_id>\d+)/$', views.classify, name='classify'),
)
