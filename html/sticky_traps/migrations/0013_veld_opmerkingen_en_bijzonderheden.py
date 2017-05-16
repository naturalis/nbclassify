# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('sticky_traps', '0012_auto_20170512_1249'),
    ]

    operations = [
        migrations.AddField(
            model_name='veld',
            name='Opmerkingen_en_bijzonderheden',
            field=models.TextField(null=True),
        ),
    ]
