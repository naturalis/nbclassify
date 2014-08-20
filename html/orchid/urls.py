from django.conf.urls import patterns, url

from orchid import views

urlpatterns = patterns('',
    url(r'^$', views.home, name='home'),
    url(r'^upload/$', views.upload, name='upload'),
    url(r'^result/$', views.result, name='result'),
    url(r'^exit/$', views.exit, name='exit'),
    url(r'^cleanup/$', views.cleanup, name='cleanup'),

    url(r'^accounts/login/$', views.login, name='auth_login'),
    url(r'^accounts/auth/$', views.auth_view, name='auth'),
    url(r'^accounts/logout/$', views.logout, name='auth_logout'),
    url(r'^accounts/invalid/$', views.invalid_login, name='auth_invalid'),
)
