# Generated by Django 5.1.5 on 2025-03-23 07:38

import django.core.validators
import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0012_lostpet'),
    ]

    operations = [
        migrations.CreateModel(
            name='PetLocation',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('latitude', models.FloatField(validators=[django.core.validators.MinValueValidator(-90.0), django.core.validators.MaxValueValidator(90.0)])),
                ('longitude', models.FloatField(validators=[django.core.validators.MinValueValidator(-180.0), django.core.validators.MaxValueValidator(180.0)])),
                ('status', models.CharField(choices=[('lost', 'Lost'), ('found', 'Found'), ('resolved', 'Resolved')], default='lost', max_length=10)),
                ('description', models.TextField(blank=True)),
                ('reported_at', models.DateTimeField(auto_now_add=True)),
                ('resolved_at', models.DateTimeField(blank=True, null=True)),
                ('contact_name', models.CharField(blank=True, max_length=100)),
                ('contact_phone', models.CharField(blank=True, max_length=20)),
                ('contact_email', models.EmailField(blank=True, max_length=254)),
                ('last_seen_date', models.DateField(blank=True, null=True)),
                ('last_seen_time', models.TimeField(blank=True, null=True)),
                ('pet', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='locations', to='accounts.pet')),
            ],
            options={
                'ordering': ['-reported_at'],
            },
        ),
        migrations.DeleteModel(
            name='LostPet',
        ),
        migrations.AddIndex(
            model_name='petlocation',
            index=models.Index(fields=['status'], name='accounts_pe_status_65be1e_idx'),
        ),
        migrations.AddIndex(
            model_name='petlocation',
            index=models.Index(fields=['pet'], name='accounts_pe_pet_id_e7e7fa_idx'),
        ),
        migrations.AddIndex(
            model_name='petlocation',
            index=models.Index(fields=['reported_at'], name='accounts_pe_reporte_3c5e5f_idx'),
        ),
    ]
