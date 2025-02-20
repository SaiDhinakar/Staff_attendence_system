from django.contrib import admin
from .models import Attendance, Employee

# Register your models here.
admin.site.register(Employee)
admin.site.register(Attendance)

# Customize admin panel styling
admin.site.site_header = 'Staff Face Recognition Admin Panel'
admin.site.site_title = 'Staff Face Recognition'
admin.site.index_title = 'Administration'


