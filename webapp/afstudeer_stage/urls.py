from django.conf.urls import patterns, include, url

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = patterns('',
    # Examples:
    url(r'^welcome/$', 'orchid.views.welcome'),
    url(r'^upload/$', 'orchid.views.upload'),
    url(r'^upload_success/$', 'orchid.views.upload_success'),
    # url(r'^$', 'afstudeer_stage.views.home', name='home'),
    # url(r'^afstudeer_stage/', include('afstudeer_stage.foo.urls')),

    # Uncomment the admin/doc line below to enable admin documentation:
    # url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    # url(r'^admin/', include(admin.site.urls)),
)
