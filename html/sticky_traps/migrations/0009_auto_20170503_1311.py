# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('sticky_traps', '0008_veld_opgeslagen'),
    ]

    operations = [
        migrations.AlterField(
            model_name='veld',
            name='Veld_nummer',
            field=models.CharField(max_length=50),
        ),
    ]
