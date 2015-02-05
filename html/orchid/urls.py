from django.conf.urls import patterns, include, url
from rest_framework import routers

from orchid import views


router = routers.DefaultRouter()
router.register(r'photos', views.PhotoViewSet)
router.register(r'identities', views.IdentityViewSet)

urlpatterns = patterns('',
    url(r'^$', views.home, name='home'),
    url(r'^photo/(?P<photo_id>\d+)/$', views.photo, name='photo'),
    url(r'^photo/(?P<photo_id>\d+)/identify/$', views.identify, name='identify'),
    url(r'^photo/(?P<photo_id>\d+)/identity/$', views.photo_identity, name='identity'),
    url(r'^photo/(?P<photo_id>\d+)/delete/$', views.delete_photo, name='delete_photo'),
    url(r'^library/$', views.my_photos, name='library'),
    url(r'^session_data\.json$', views.json_get_session_data, name='session_data'),
    url(r'^eol_species_info/(?P<query>[\w\s\+-]+)/$', views.eol_orchid_species_info, name='eol_species_info'),
    url(r'^orchid\.js$', views.javascript, name='js'),

    # Automatic URL routing for the API.
    url(r'^api/', include(router.urls)),
    # Custom URLs for the API.
    url(r'^api/identities/(?P<pk>[0-9]+)/info/$', views.IdentityInfoView.as_view(), name='id_info'),
    # Include login URLs for the browsable API.
    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework'))
)
