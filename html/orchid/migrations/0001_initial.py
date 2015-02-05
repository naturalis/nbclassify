# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations
import orchid.models


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Identity',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('genus', models.CharField(max_length=50)),
                ('section', models.CharField(max_length=50, null=True, blank=True)),
                ('species', models.CharField(max_length=50, null=True, blank=True)),
                ('error', models.FloatField()),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='Photo',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('image', models.ImageField(upload_to=orchid.models.get_image_path)),
                ('roi', models.CharField(max_length=30, null=True, blank=True)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.AddField(
            model_name='identity',
            name='photo',
            field=models.ForeignKey(related_name='identities', to='orchid.Photo'),
            preserve_default=True,
        ),
    ]
