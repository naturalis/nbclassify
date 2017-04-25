# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('sticky_traps', '0005_auto_20170425_0749'),
    ]

    operations = [
        migrations.AddField(
            model_name='photo',
            name='veldnr',
            field=models.PositiveIntegerField(default=27),
            preserve_default=False,
        ),
    ]
