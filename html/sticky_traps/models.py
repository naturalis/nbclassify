import hashlib
import os.path
import time
import datetime

from django.db import models
from django.db.models.signals import post_delete
from django.dispatch.dispatcher import receiver

def get_image_path(instance, filename):
    """Return the path for an uploaded image.

    Uploaded images are placed in ``sticky_traps/uploads/`` and the file is renamed to
    the file's MD5 sum.
    """
    hasher = hashlib.md5()
    buf = instance.foto.read()
    hasher.update(buf)
    parts = os.path.splitext(filename)
    filename_ = "%s%s" % (hasher.hexdigest()[:10], parts[1])
    path = "sticky_traps/uploads/%%Y/%%m/%%d/%s" % (filename_,)
    return time.strftime(path)

    def __unicode__(self):
        if self.species:
            return "%s %s" % (self.genus, self.species)
        else:
            return self.genus

class Veld(models.Model):
    Opgeslagen = models.BooleanField(default=False)
    Veld_nummer = models.CharField(max_length=50)
    Breedtegraad = models.DecimalField(decimal_places=5, max_digits=10)
    Lengtegraad = models.DecimalField(decimal_places=5, max_digits=10)
    Beheer_type = models.CharField(max_length=50)
    Plaatsings_datum = models.DateField(default=datetime.datetime.now)
    Verwijderings_datum = models.DateField(default=datetime.datetime.now)
    MIDDEN = "M"
    RAND = "R"
    SLOOTKANT = "S"
    Locatie_binnen_veld_keuzes = (
        (MIDDEN, "Midden van het veld"),
        (RAND, "Aan de rand van het veld"),
        (SLOOTKANT, "Bij een sloot")
        )
    Locatie_binnen_veld = models.CharField(
        max_length=30,
        choices=Locatie_binnen_veld_keuzes,
        blank=False
        )
    Beweiding = models.BooleanField(default=False)
    Maaien = models.BooleanField(default=False)
    Minimale_hoogte_gras = models.DecimalField(decimal_places=2, max_digits=5)
    Maximale_hoogte_gras = models.DecimalField(decimal_places=2, max_digits=5)
    WEINIG = "Weinig biodiversiteit"
    GEMIDDELD = "Gemiddelde biodiversiteit"
    VEEL = "Hoge bidodiversiteit"
    Hoeveelheid_biodiversiteit_keuzes = (
        (WEINIG, "Weinig biodiversiteit"),
        (GEMIDDELD, "Gemiddelde biodiversiteit"),
        (VEEL, "Hoge biodiversiteit")
        )
    Hoeveelheid_biodiversiteit = models.CharField(
    max_length=30,
    choices=Hoeveelheid_biodiversiteit_keuzes,
    blank=False
    )

    def __str__(self):
        return str(self.Veld_nummer)

class Photo(models.Model):
    """Model for uploaded photos."""
    foto = models.ImageField(upload_to=get_image_path)
    veld = models.ForeignKey(Veld, null=True)
    veldnr = models.PositiveIntegerField(null=True)
    code = models.CharField(max_length=30)


    def __unicode__(self):
        return self.file_name()

    def file_name(self):
        return os.path.basename(self.foto.name)

    def image_tag(self):
        if self.image:
            return u'<img src="%s" width="250px" />' % (self.image.url)
        else:
            return "(No photo)"
    image_tag.short_description = 'Thumbnail'
    image_tag.allow_tags = True


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
