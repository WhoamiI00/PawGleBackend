# Generated by Django 5.1.5 on 2025-01-25 13:56

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0002_rename_additional_info_pet_additionalinfo_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='pet',
            name='animal_id',
            field=models.CharField(default='ANI0000', max_length=20, unique=True),
        ),
        migrations.AddField(
            model_name='pet',
            name='features',
            field=models.JSONField(default=list),
        ),
        migrations.AddField(
            model_name='pet',
            name='images',
            field=models.JSONField(default=list),
        ),
        migrations.AddField(
            model_name='pet',
            name='registered_at',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
    ]
