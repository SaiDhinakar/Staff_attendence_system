from django.db import models
from django.utils import timezone

class Employee(models.Model):
    emp_id = models.CharField(max_length=10, primary_key=True)
    emp_name = models.CharField(max_length=100)
    department = models.CharField(max_length=50)
    
    def __str__(self):
        return f"{self.emp_id} - {self.emp_name}"

class Attendance(models.Model):
    STATUS_CHOICES = [
        ('present', 'Present'),
        ('absent', 'Absent')
    ]
    
    id = models.AutoField(primary_key=True)
    emp = models.ForeignKey(Employee, on_delete=models.CASCADE)
    date = models.DateField(default=timezone.now)
    time_in = models.TimeField(null=True, blank=True)
    time_out = models.TimeField(null=True, blank=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES)
    
    class Meta:
        ordering = ['-date', 'emp__emp_name']
    
    def __str__(self):
        return f"{self.emp.emp_name} - {self.date}"
