# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('sticky_traps', '0011_auto_20170511_1302'),
    ]

    operations = [
        migrations.RenameField(
            model_name='photo',
            old_name='code',
            new_name='Val_nummer',
        ),
        migrations.RenameField(
            model_name='veld',
            old_name='Veld_nummer',
            new_name='Veld_identificatie_code',
        ),
        migrations.RemoveField(
            model_name='veld',
            name='Locatie_binnen_veld',
        ),
        migrations.RemoveField(
            model_name='veld',
            name='Verwijderings_datum',
        ),
        migrations.AlterField(
            model_name='veld',
            name='Beheer_type',
            field=models.CharField(max_length=50, null=True),
        ),
    ]
