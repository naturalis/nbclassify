import os.path

from django.db import models
from django.db.models.signals import post_delete
from django.dispatch.dispatcher import receiver

class Photo(models.Model):
    """Model for uploaded photos."""
    image = models.ImageField(upload_to='orchid/uploads')
    roi = models.CharField(max_length=30, null=True, blank=True)

    def __unicode__(self):
        return self.file_name()

    def file_name(self):
        return os.path.basename(self.image.name)

    def image_tag(self):
        if self.image:
            return u'<img src="%s" width="250px" />' % (self.image.url)
        else:
            return "(No photo)"
    image_tag.short_description = 'Thumbnail'
    image_tag.allow_tags = True

class Identity(models.Model):
    photo = models.ForeignKey(Photo)
    genus = models.CharField(max_length=50)
    section = models.CharField(max_length=50, null=True, blank=True)
    species = models.CharField(max_length=50, null=True, blank=True)
    error = models.FloatField()

    def __unicode__(self):
        if self.species:
            return "%s %s" % (self.genus, self.species)
        else:
            return self.genus

@receiver(post_delete, sender=Photo)
def photo_delete_hook(sender, instance, **kwargs):
    """Delete file associated with Photo instance.

    Receive the ``post_delete`` signal and delete the file associated with the
    model instance. This removes the associated file when the model instance
    is removed from the Django Admin.
    """
    # Pass False so ImageField doesn't save the model.
    if instance.image:
        instance.image.delete(False)
