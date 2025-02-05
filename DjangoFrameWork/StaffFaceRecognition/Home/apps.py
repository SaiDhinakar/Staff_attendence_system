from django.apps import AppConfig
import subprocess


class HomeConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'Home'

    def ready(self):
        if not hasattr(self, 'fastapi_process'):
            self.fastapi_process = subprocess.Popen(['uvicorn', 'backend.face_recognition:app', '--host', '0.0.0.0', '--port', '5000'])