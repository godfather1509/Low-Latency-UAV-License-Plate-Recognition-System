from django.db import models

# Create your models here.


class videoSave(models.Model):
    video_file=models.FileField(upload_to='Videos')
    date_time=models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.date_time)