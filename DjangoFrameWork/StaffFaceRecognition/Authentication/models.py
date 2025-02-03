from django.db import models
from django.contrib.auth.models import User

class StaffUser(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    employee_id = models.CharField(max_length=20, unique=True)
    department = models.CharField(max_length=100)
    contact_number = models.CharField(max_length=15)
    
    def __str__(self):
        return f"{self.user.username} - {self.employee_id}"