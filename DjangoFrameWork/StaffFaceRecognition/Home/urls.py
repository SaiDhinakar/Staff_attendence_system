from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    # path('settings/', views.settings_view, name='settings'),
    path('manage-employees/', views.manage_employees, name='manage_employees'),
    path('report/', views.report_view, name='report'),
    path('export-report/', views.export_report, name='export_report'),
]