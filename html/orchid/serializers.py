from django.contrib.auth.models import User, Group
from rest_framework import serializers

from orchid.models import Photo, Identity

class PhotoSerializer(serializers.ModelSerializer):
    identities = serializers.PrimaryKeyRelatedField(many=True, read_only=True)

    class Meta:
        model = Photo
        fields = ('id','image','roi','identities')

class IdentitySerializer(serializers.ModelSerializer):
    class Meta:
        model = Identity
        fields = ('id','photo','genus','section','species','error')
