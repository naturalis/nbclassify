import hashlib
import os.path
import time

from django.db import models
from django.db.models.signals import post_delete
from django.dispatch.dispatcher import receiver

def get_image_path(instance, filename):
    """Return the path for an uploaded image.

    Uploaded images are placed in ``orchid/uploads/`` and the file is renamed to
    the file's MD5 sum.
    """
    hasher = hashlib.md5()
    buf = instance.image.read()
    hasher.update(buf)
    parts = os.path.splitext(filename)
    filename_ = "%s%s" % (hasher.hexdigest(), parts[1])
    path = "orchid/uploads/%%Y/%%m/%%d/%s" % (filename_,)
    return time.strftime(path)

class Photo(models.Model):
    """Model for uploaded photos."""
    image = models.ImageField(upload_to=get_image_path)
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
    photo = models.ForeignKey(Photo, related_name="identities")
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
