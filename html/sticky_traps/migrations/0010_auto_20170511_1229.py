# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import geoposition.fields


class Migration(migrations.Migration):

    dependencies = [
        ('sticky_traps', '0009_auto_20170503_1311'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='veld',
            name='Breedtegraad',
        ),
        migrations.RemoveField(
            model_name='veld',
            name='Lengtegraad',
        ),
        migrations.AddField(
            model_name='veld',
            name='Locatie',
            field=geoposition.fields.GeopositionField(default=None, max_length=42),
        ),
    ]
