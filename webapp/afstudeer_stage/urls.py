# Import the required modules
from django.conf.urls import patterns, include, url

# Uncomment the next two lines to enable the admin:
from django.contrib import admin
admin.autodiscover()

urlpatterns = patterns('',
    # The first two urls display both the welcome page.
    # The fist one is for direct use (e.g. for local 127.0.0.1:8000 will
    # display the welcome page)
    # The second is used in the exit view, because it is not posible to
    # redirect to an empty url
    url(r'^$', 'orchid.views.welcome'),
    url(r'^welcome/$', 'orchid.views.welcome'),
    url(r'^upload/$', 'orchid.views.upload'),
    # url for the result page, using the result view
    url(r'^result/$', 'orchid.views.result'),
    # url for the exit page, using the exit view
    url(r'^exit/$', 'orchid.views.exit'),
    # url(r'^$', 'afstudeer_stage.views.home', name='home'),
    # url(r'^afstudeer_stage/', include('afstudeer_stage.foo.urls')),

    # Uncomment the admin/doc line below to enable admin documentation:
    # url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    url(r'^admin/', include(admin.site.urls)),
    url(r'^admin/remove', 'orchid.views.remove'),
    
    # urls for logging in end out
    url(r'^accounts/login/$', 'orchid.views.login'),
    url(r'^accounts/auth/$', 'orchid.views.auth_view'),
    url(r'^accounts/logout/$', 'orchid.views.logout'),
    url(r'^accounts/invalid/$', 'orchid.views.invalid_login'),
    
    # When a picture is removed and the software search for it, go to this url
    url(r'^sorry/$', 'orchid.views.sorry'),
)
