from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    path('api/get_env/', views.get_env_values),
    path('manage-employees/', views.manage_employees, name='manage_employees'),
    path('report/', views.report_view, name='report'),
    path('export-report/', views.export_report, name='export_report'),
    path('store-embeddings/', views.store_embeddings, name='store_embeddings'),
    path('employee/<str:employee_id>/', views.employee_detail, name='employee_detail'),
    path('debug-employee/<int:employee_id>/', views.debug_employee_detail, name='debug_employee_detail'),
    path('get_attendace/',views.get_attendance, name='get_attendace'),
]