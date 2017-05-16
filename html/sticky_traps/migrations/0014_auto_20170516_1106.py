# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import sticky_traps.models


class Migration(migrations.Migration):

    dependencies = [
        ('sticky_traps', '0013_veld_opmerkingen_en_bijzonderheden'),
    ]

    operations = [
        migrations.AlterField(
            model_name='photo',
            name='Val_nummer',
            field=models.CharField(max_length=30, null=True),
        ),
        migrations.AlterField(
            model_name='photo',
            name='foto',
            field=models.ImageField(null=True, upload_to=sticky_traps.models.get_image_path),
        ),
    ]
