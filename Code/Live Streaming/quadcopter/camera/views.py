from django.shortcuts import render
from rest_framework.viewsets import ModelViewSet
from .serializer import videoSerializer
from .models import videoSave

class saveVideo(ModelViewSet):
    serializer_class=videoSerializer
    queryset=videoSave.objects.all()


# Create your views here.
