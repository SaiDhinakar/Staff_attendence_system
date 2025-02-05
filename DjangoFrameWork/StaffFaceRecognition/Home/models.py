from django.db import models
from django.utils import timezone
from datetime import datetime
from django.shortcuts import render

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
    time_list = models.TextField(blank=True, null=True)  # Store times as comma-separated string
    
    def get_times(self):
        if not self.time_list:
            return []
        return self.time_list.split(';')
    
    def get_in_time(self):
        times = self.get_times()
        return times[0] if times else None
    
    def get_out_time(self):
        times = self.get_times()
        return times[-1] if times else None
    
    def get_working_hours(self):
        times = self.get_times()
        if len(times) < 2:
            return 0
            
        in_time = datetime.strptime(times[0], '%H:%M')
        out_time = datetime.strptime(times[-1], '%H:%M')
        diff = out_time - in_time
        return round(diff.total_seconds() / 3600, 2)
    
    def get_status(self):
        times = self.get_times()
        if not times:
            return 'Absent'
        if len(times) == 1:
            return 'Present (No Out Time)'
        return 'Present'
    
    class Meta:
        ordering = ['-date', 'emp__emp_name']
    
    def __str__(self):
        return f"{self.emp.emp_name} - {self.date}"

def process_attendance(request, emp_id):
    current_time = timezone.now().strftime('%H:%M')
    today = timezone.now().date()
    
    attendance, created = Attendance.objects.get_or_create(
        emp_id=emp_id,
        date=today
    )
    
    times = attendance.get_times()
    if times:
        times.append(current_time)
        attendance.time_list = ';'.join(times)
    else:
        attendance.time_list = current_time
        
    attendance.save()

def report_view(request):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    
    attendance_query = Attendance.objects.select_related('emp')
    
    if start_date and end_date:
        attendance_query = attendance_query.filter(
            date__range=[start_date, end_date]
        )
    
    attendance_data = [{
        'date': att.date,
        'emp_id': att.emp.emp_id,
        'emp_name': att.emp.emp_name,
        'department': att.emp.department,
        'in_time': att.get_in_time(),
        'out_time': att.get_out_time(),
        'working_hours': att.get_working_hours(),
        'status': att.get_status()
    } for att in attendance_query]
    
    return render(request, 'report.html', {'attendance_data': attendance_data})
