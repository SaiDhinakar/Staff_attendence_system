from django.db import models
from django.utils import timezone
from datetime import datetime

class Employee(models.Model):
    emp_id = models.CharField(max_length=10, primary_key=True)
    emp_name = models.CharField(max_length=100)
    department = models.CharField(max_length=50)
    
    def __str__(self):
        return f"{self.emp_id} - {self.emp_name}"

class Attendance(models.Model):
    id = models.AutoField(primary_key=True)
    emp = models.ForeignKey(Employee, on_delete=models.CASCADE)
    date = models.DateField(default=timezone.now)
    time_in_list = models.TextField(blank=True, null=True)
    time_out_list = models.TextField(blank=True, null=True)
    
    def get_in_time(self):
        return self.time_in_list.split(';') if self.time_in_list else []
        
    def get_out_time(self):
        return self.time_out_list.split(';') if self.time_out_list else []
        
    def get_working_hours(self):
        in_times = self.get_in_time()
        out_times = self.get_out_time()
        total_hours = 0
        
        for i in range(min(len(in_times), len(out_times))):
            try:
                in_dt = datetime.strptime(in_times[i].strip(), '%H:%M')
                out_dt = datetime.strptime(out_times[i].strip(), '%H:%M')
                diff = out_dt - in_dt
                total_hours += diff.seconds / 3600
            except ValueError:
                continue
                
        return round(total_hours, 2)
    
    def get_status(self):
        in_times = self.get_in_time()
        if not in_times:
            return 'Absent'
        out_times = self.get_out_time()
        if len(in_times) > len(out_times):
            return 'Present (No Out Time)'
        return 'Present'
    
    class Meta:
        ordering = ['-date', 'emp__emp_name']
    
    def __str__(self):
        return f"{self.emp.emp_name} - {self.date}"