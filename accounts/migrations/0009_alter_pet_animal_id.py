# Generated by Django 5.1.5 on 2025-01-30 07:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0008_alter_pet_animal_id_alter_pet_category'),
    ]

    operations = [
        migrations.AlterField(
            model_name='pet',
            name='animal_id',
            field=models.CharField(editable=False, max_length=20, unique=True),
        ),
    ]
