from django.conf.urls import patterns, include, url
from rest_framework import routers

from sticky_traps import views

# Create a router and register the viewsets with it.
router = routers.DefaultRouter()
router.register(r'photos', views.PhotoViewSet)
router.register(r'identities', views.IdentityViewSet)

urlpatterns = patterns('',
    url(r'^$', views.home, name='home'),
    url(r'^photo/(?P<photo_id>\d+)/$', views.photo, name='photo'),
    url(r'^photo/(?P<photo_id>\d+)/identities/$', views.photo_identities, name='photo-identities'),
    #url(r'^identity/(?P<pk>\d+)/eol/$', views.identity_eol_info, name='identity-eol'),
    url(r'^library/$', views.my_photos, name='library'),
    url(r'^session_data\.json$', views.json_get_session_data, name='session_data'),
    url(r'^orchid\.js$', views.javascript, name='js'),

    # The API URLs are determined automatically by the router.
    url(r'^api/', include(router.urls, namespace='api')),
)
