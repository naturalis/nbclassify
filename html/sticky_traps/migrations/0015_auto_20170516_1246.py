# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import datetime
import geoposition.fields
import sticky_traps.models


class Migration(migrations.Migration):

    dependencies = [
        ('sticky_traps', '0014_auto_20170516_1106'),
    ]

    operations = [
        migrations.AddField(
            model_name='veld',
            name='gemiddeld_oppervlak_over_veld',
            field=models.FloatField(null=True),
        ),
        migrations.AddField(
            model_name='veld',
            name='variance',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='photo',
            name='Val_nummer',
            field=models.CharField(help_text=b'Het laatste deel van de code op de val. Als op de val bijvoorbeeld Koning-UG-03 staat, vul dan 03 in.', max_length=30, null=True),
        ),
        migrations.AlterField(
            model_name='photo',
            name='foto',
            field=models.ImageField(help_text=b'selecteer de goede afbeelding vanaf uw computer of smartphone', null=True, upload_to=sticky_traps.models.get_image_path),
        ),
        migrations.AlterField(
            model_name='veld',
            name='Beheer_type',
            field=models.CharField(help_text=b'Het type beheer wat word gebruikt zoals uitgesteld maaien of legselbescherming.', max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='veld',
            name='Beweiding',
            field=models.BooleanField(default=False, help_text=b'Vink dit vakje aan als er beweid word.'),
        ),
        migrations.AlterField(
            model_name='veld',
            name='Hoeveelheid_biodiversiteit',
            field=models.CharField(help_text=b'geef een schatting van de hoeveelheid verschillende kruiden in dit veld; veel, gemiddeld, of weinig', max_length=30, choices=[(b'Weinig biodiversiteit', b'Weinig biodiversiteit'), (b'Gemiddelde biodiversiteit', b'Gemiddelde biodiversiteit'), (b'Hoge bidodiversiteit', b'Hoge biodiversiteit')]),
        ),
        migrations.AlterField(
            model_name='veld',
            name='Locatie',
            field=geoposition.fields.GeopositionField(help_text=b'Zoek op de plaatsnaam van de boerderij en versleep de marker vervolgens naar het midden van het veld waar de vallen zijn uitgezet.', max_length=42),
        ),
        migrations.AlterField(
            model_name='veld',
            name='Maaien',
            field=models.BooleanField(default=False, help_text=b'Vink dit vakje aan als er op dit perceel gemaaid word'),
        ),
        migrations.AlterField(
            model_name='veld',
            name='Maximale_hoogte_gras',
            field=models.DecimalField(help_text=b'de geschatte maximale hoogte van het gras op dit perceel in centimeters', max_digits=5, decimal_places=2),
        ),
        migrations.AlterField(
            model_name='veld',
            name='Minimale_hoogte_gras',
            field=models.DecimalField(help_text=b'de geschatte minimale hoogte van het gras op dit perceel in centimeters', max_digits=5, decimal_places=2),
        ),
        migrations.AlterField(
            model_name='veld',
            name='Opmerkingen_en_bijzonderheden',
            field=models.TextField(help_text=b'geef eventuele bijzonderheden of opmerkingen hier aan', null=True),
        ),
        migrations.AlterField(
            model_name='veld',
            name='Plaatsings_datum',
            field=models.DateField(default=datetime.datetime.now, help_text=b'De datum waarop de vallen zijn uitgezet'),
        ),
        migrations.AlterField(
            model_name='veld',
            name='Veld_identificatie_code',
            field=models.CharField(help_text=b'Het eerste deel van de code die op de val staat. Als op de val bijvoorbeeld Koning-UG-03 staat, schrijf dan Koning-UG op.', max_length=50),
        ),
    ]
