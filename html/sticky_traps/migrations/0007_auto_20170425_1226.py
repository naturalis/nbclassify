# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('sticky_traps', '0006_photo_veldnr'),
    ]

    operations = [
        migrations.AlterField(
            model_name='photo',
            name='veldnr',
            field=models.PositiveIntegerField(null=True),
        ),
    ]
