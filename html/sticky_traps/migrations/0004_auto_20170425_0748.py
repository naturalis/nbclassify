# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('sticky_traps', '0003_auto_20170421_1743'),
    ]

    operations = [
        migrations.AlterField(
            model_name='photo',
            name='veld',
            field=models.ForeignKey(to='sticky_traps.Veld'),
        ),
    ]
