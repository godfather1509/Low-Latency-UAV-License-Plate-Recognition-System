import os
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
from django.core.asgi import get_asgi_application
import camera.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'quadcopter.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(camera.routing.websocket_urlpatterns)
    ),
})
