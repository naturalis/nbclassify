# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import geoposition.fields


class Migration(migrations.Migration):

    dependencies = [
        ('sticky_traps', '0010_auto_20170511_1229'),
    ]

    operations = [
        migrations.AlterField(
            model_name='veld',
            name='Locatie',
            field=geoposition.fields.GeopositionField(max_length=42),
        ),
    ]
