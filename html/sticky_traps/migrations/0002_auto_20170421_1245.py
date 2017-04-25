# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('sticky_traps', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Veld',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('Veld_nummer', models.PositiveIntegerField()),
                ('Breedtegraad', models.DecimalField(max_digits=10, decimal_places=5)),
                ('Lengtegraad', models.DecimalField(max_digits=10, decimal_places=5)),
                ('Beheer_type', models.CharField(max_length=50)),
                ('Plaatsings_datum', models.DateField()),
                ('Verwijderings_datum', models.DateField()),
                ('Locatie_binnen_veld', models.CharField(max_length=30, choices=[(b'M', b'Midden van het veld'), (b'R', b'Aan de rand van het veld'), (b'S', b'Bij een sloot')])),
                ('Beweiding', models.BooleanField(default=False)),
                ('Maaien', models.BooleanField(default=False)),
                ('Minimale_hoogte_gras', models.DecimalField(max_digits=5, decimal_places=2)),
                ('Maximale_hoogte_gras', models.DecimalField(max_digits=5, decimal_places=2)),
                ('Hoeveelheid_biodiversiteit', models.CharField(max_length=30, choices=[(b'Weinig biodiversiteit', b'Weinig biodiversiteit'), (b'Gemiddelde biodiversiteit', b'Gemiddelde biodiversiteit'), (b'Hoge bidodiversiteit', b'Hoge biodiversiteit')])),
            ],
        ),
        migrations.RemoveField(
            model_name='identity',
            name='photo',
        ),
        migrations.RenameField(
            model_name='photo',
            old_name='image',
            new_name='foto',
        ),
        migrations.RemoveField(
            model_name='photo',
            name='roi',
        ),
        migrations.DeleteModel(
            name='Identity',
        ),
        migrations.AddField(
            model_name='photo',
            name='veld',
            field=models.ForeignKey(default=None, to='sticky_traps.Veld'),
        ),
    ]
