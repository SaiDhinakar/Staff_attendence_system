from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    # path('settings/', views.settings_view, name='settings'),
    # path('settings/update-user/<int:user_id>/', views.update_user, name='update_user'),
    path('report/', views.report_view, name='report'),
    path('export-report/', views.export_report, name='export_report'),
]