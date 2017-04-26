# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('sticky_traps', '0007_auto_20170425_1226'),
    ]

    operations = [
        migrations.AddField(
            model_name='veld',
            name='Opgeslagen',
            field=models.BooleanField(default=False),
        ),
    ]
