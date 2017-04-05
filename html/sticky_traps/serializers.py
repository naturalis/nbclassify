from django.contrib.auth.models import User, Group
from rest_framework import serializers

from orchid.models import Photo, Identity

class PhotoSerializer(serializers.ModelSerializer):
    identities = serializers.PrimaryKeyRelatedField(many=True, read_only=True)

    class Meta:
        model = Photo
        fields = ('id','image','roi','identities')

    def validate_roi(self, value):
        """Validate the ROI field."""
        try:
            roi = value.split(',')
            roi = [int(x) for x in roi]

            assert len(roi) == 4, \
                "The ROI must have the format `x,y,width,height`"
            assert roi[0] >= 0, \
                "ROI x value out of bounds"
            assert roi[1] >= 0, \
                "ROI y value out of bounds"
            assert roi[2] >= 1, \
                "ROI width value out of bounds"
            assert roi[3] >= 1, \
                "ROI height value out of bounds"
        except:
            raise serializers.ValidationError("Must be of the format `x,y,width,height`")

        return value

class IdentitySerializer(serializers.ModelSerializer):
    class Meta:
        model = Identity
        fields = ('id','photo','genus','section','species','error')
        read_only_fields = fields
