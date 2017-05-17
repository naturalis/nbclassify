# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('sticky_traps', '0017_auto_20170517_0839'),
    ]

    operations = [
        migrations.AddField(
            model_name='photo',
            name='unieke_veld_code',
            field=models.IntegerField(null=True),
        ),
    ]
