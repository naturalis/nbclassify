# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import datetime


class Migration(migrations.Migration):

    dependencies = [
        ('sticky_traps', '0002_auto_20170421_1245'),
    ]

    operations = [
        migrations.AddField(
            model_name='photo',
            name='code',
            field=models.CharField(default='Koning-KG-01a', max_length=30),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='veld',
            name='Plaatsings_datum',
            field=models.DateField(default=datetime.datetime.now),
        ),
        migrations.AlterField(
            model_name='veld',
            name='Verwijderings_datum',
            field=models.DateField(default=datetime.datetime.now),
        ),
    ]
