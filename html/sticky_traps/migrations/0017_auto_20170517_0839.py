# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('sticky_traps', '0016_photo_datum'),
    ]

    operations = [
        migrations.AlterField(
            model_name='photo',
            name='veldnr',
            field=models.CharField(max_length=50, null=True),
        ),
    ]
