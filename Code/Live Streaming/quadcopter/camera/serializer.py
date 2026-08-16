from rest_framework import serializers
from .models import videoSave
from rest_framework.serializers import ModelSerializer


class videoSerializer(ModelSerializer):
    class Meta:
        model=videoSave
        fields='__all__'
