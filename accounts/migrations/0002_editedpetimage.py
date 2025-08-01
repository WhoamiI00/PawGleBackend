# Generated by Django 5.1.5 on 2025-04-18 07:07

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='EditedPetImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('edited_image', models.ImageField(blank=True, null=True, upload_to='edited_pet_images/')),
                ('edit_metadata', models.JSONField(default=dict, help_text='Stores editing parameters like filters, shapes, etc.')),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('original_image', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='original_for_edits', to='accounts.pet')),
                ('owner', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='edited_pet_images', to=settings.AUTH_USER_MODEL)),
                ('pet', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='edited_images', to='accounts.pet')),
            ],
            options={
                'ordering': ['-created_at'],
                'indexes': [models.Index(fields=['pet', 'owner'], name='accounts_ed_pet_id_c82e1a_idx')],
            },
        ),
    ]
