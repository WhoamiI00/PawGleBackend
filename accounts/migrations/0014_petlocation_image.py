# Generated by Django 5.1.5 on 2025-03-28 08:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0013_petlocation_delete_lostpet_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='petlocation',
            name='image',
            field=models.ImageField(blank=True, null=True, upload_to='pet_locations/'),
        ),
    ]
