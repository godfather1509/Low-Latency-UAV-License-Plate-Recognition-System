from django.contrib import admin
from django.urls import path,include
from camera.views import saveVideo
from rest_framework.routers import DefaultRouter


router=DefaultRouter()

router.register(r'video',saveVideo,basename="video")

urlpatterns = [
path('',include(router.urls)),
]
