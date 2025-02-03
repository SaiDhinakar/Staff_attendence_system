from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from .models import StaffUser

class StaffUserInline(admin.StackedInline):
    model = StaffUser
    can_delete = False

class CustomUserAdmin(UserAdmin):
    inlines = (StaffUserInline,)
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_staff', 'get_employee_id')
    
    def get_employee_id(self, obj):
        try:
            return obj.staffuser.employee_id
        except StaffUser.DoesNotExist:
            return None
    get_employee_id.short_description = 'Employee ID'

admin.site.unregister(User)
admin.site.register(User, CustomUserAdmin)