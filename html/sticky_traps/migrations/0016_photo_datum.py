# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('sticky_traps', '0015_auto_20170516_1246'),
    ]

    operations = [
        migrations.AddField(
            model_name='photo',
            name='datum',
            field=models.DateField(null=True),
        ),
    ]
