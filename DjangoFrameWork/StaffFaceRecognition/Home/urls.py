from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('settings/', views.settings_view, name='settings'),
]