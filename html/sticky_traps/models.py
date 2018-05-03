import hashlib
import os.path
import time
import datetime

from django.db import models
from django.db.models.signals import post_delete
from django.dispatch.dispatcher import receiver
from geoposition.fields import GeopositionField

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
    path = "%%Y/%%m/%%d/%s" % (filename_,)
    return time.strftime(path)

    def __unicode__(self):
        if self.species:
            return "%s %s" % (self.genus, self.species)
        else:
            return self.genus

class Veld(models.Model):
    Opgeslagen = models.BooleanField(default=False)
    Veld_identificatie_code = models.CharField(max_length=50, help_text="Het eerste deel van de code die op de val staat. Als op de val bijvoorbeeld Koning-UG-03 staat, schrijf dan Koning-UG op.")
    Locatie = GeopositionField(help_text="Zoek op de plaatsnaam van de boerderij en versleep de marker vervolgens naar het midden van het veld waar de vallen zijn uitgezet.")
    Plaatsings_datum = models.DateField(default=datetime.datetime.now, help_text="De datum waarop de vallen zijn uitgezet")
    GEEN = "Landbouw: geen beheerpakket"
    UM = "Landbouw: uitgesteld maaien"
    LB = "Landbouw: legselbeheer"
    KG = "Landbouw: kruidenrijk grasland"
    VH = "Natuur: vochtig hooiland"
    WP = "Natuur: weidevogel pakket"
    NO = "Weet ik niet"
    Beheer_type_keuzes = (
        (GEEN, "Landbouw: geen beheerpakket"),
        (UM, "Landbouw: uitgesteld maaien"),
        (LB, "Landbouw: legselbeheer"),
        (KG, "Landbouw: kruidenrijk grasland"),
        (VH, "Natuur: vochtig hooiland"),
        (WP, "Natuur: weidevogel pakket"),
        (NO, "Weet ik niet")
        )
    Beheer_type = models.CharField(max_length=50, choices=Beheer_type_keuzes, blank=False, help_text="Het type beheer wat word gebruikt in het weiland.")
    VAST = "Vaste mest"
    RUIG = "Ruige mest"
    NO = "Weet ik niet"
    Bemesting_keuzes = (
        (VAST, "Vaste mest"),
        (RUIG, "Ruige mest"),
        (NO, "Weet ik niet")
        )
    Bemesting = models.CharField(max_length=30, choices=Bemesting_keuzes, blank=False, help_text="Geef het type bemesting aan.")
    Beweiding = models.BooleanField(default=False, help_text="Vink dit vakje aan als er beweid word.")
    Maaien = models.BooleanField(default=False, help_text="Vink dit vakje aan als er op dit perceel gemaaid word")
    Minimale_hoogte_gras = models.DecimalField(decimal_places=2, max_digits=5, help_text="de geschatte minimale hoogte van het gras op dit perceel in centimeters")
    Maximale_hoogte_gras = models.DecimalField(decimal_places=2, max_digits=5, help_text="de geschatte maximale hoogte van het gras op dit perceel in centimeters")
    WEINIG = "Weinig biodiversiteit"
    GEMIDDELD = "Gemiddelde biodiversiteit"
    VEEL = "Hoge bidodiversiteit"
    Hoeveelheid_biodiversiteit_keuzes = (
        (WEINIG, "Weinig biodiversiteit"),
        (GEMIDDELD, "Gemiddelde biodiversiteit"),
        (VEEL, "Hoge biodiversiteit")
        )
    Hoeveelheid_biodiversiteit = models.CharField(max_length=30, choices=Hoeveelheid_biodiversiteit_keuzes, blank=False, help_text="geef een schatting van de hoeveelheid verschillende kruiden in dit veld; veel, gemiddeld, of weinig")
    Opmerkingen_en_bijzonderheden = models.TextField(max_length=100, help_text="geef eventuele bijzonderheden of opmerkingen hier aan")
    gemiddeld_oppervlak_over_veld = models.FloatField(null=True)
    variance = models.FloatField(null=True)

    def __str__(self):
        return str(self.Veld_identificatie_code)

class Photo(models.Model):
    """Model for uploaded photos."""
    veld = models.ForeignKey(Veld, null=True)
    veldnr = models.CharField(null=True, max_length=50)
    unieke_veld_code = models.IntegerField(null=True)
    Val_nummer = models.CharField(max_length=30, null=True, help_text="Het laatste deel van de code op de val. Als op de val bijvoorbeeld Koning-UG-03 staat, vul dan 03 in.")
    foto = models.ImageField(upload_to=get_image_path, null=True, help_text="selecteer de goede afbeelding vanaf uw computer of smartphone")
    datum = models.DateField(null=True)


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
    try:
        if instance.image:
            instance.image.delete(False)
        else:
            pass
    except AttributeError:
        pass