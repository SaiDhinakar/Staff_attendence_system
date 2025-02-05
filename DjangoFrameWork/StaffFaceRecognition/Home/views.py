from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import HttpResponse
from django.utils import timezone
from .models import Employee, Attendance
import logging
import csv

logger = logging.getLogger(__name__)

def is_superuser(user):
    return user.is_superuser

@login_required
def home_view(request):
    # Get current date
    today = timezone.now().date()
    
    # Get all employees
    total_staff = Employee.objects.count()
    
    # Get today's attendance
    today_attendance = Attendance.objects.filter(date=today).select_related('emp')
    
    # Calculate statistics
    present_count = today_attendance.filter(status='present').count()
    absent_count = total_staff - present_count
    
    context = {
        'user': request.user,
        'is_superuser': request.user.is_superuser,
        'total_staff': total_staff,
        'present_count': present_count,
        'absent_count': absent_count,
        'attendance_data': today_attendance,
        'current_date': today
    }
    
    return render(request, 'home.html', context)

@login_required
@user_passes_test(is_superuser)
def settings_view(request):
    users = User.objects.all().exclude(username=request.user.username)
    context = {
        'users': users,
        'total_users': users.count(),
        'active_users': users.filter(is_active=True).count(),
        'staff_users': users.filter(is_staff=True).count(),
        'superusers': users.filter(is_superuser=True).count()
    }
    return render(request, 'admin_panel.html', context)


@login_required
@user_passes_test(is_superuser)
def update_user(request, user_id):
    if request.method == 'POST':
        user = User.objects.get(id=user_id)
        user.is_active = request.POST.get('is_active') == 'true'
        user.is_staff = request.POST.get('is_staff') == 'true'
        user.is_superuser = request.POST.get('is_superuser') == 'true'
        user.save()
        messages.success(request, f'User {user.username} updated successfully')
    return redirect('settings')


@login_required
def report_view(request):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    
    # Query attendance data
    attendance_query = Attendance.objects.all()
    if start_date and end_date:
        attendance_query = attendance_query.filter(
            date__range=[start_date, end_date]
        )
    
    # Get statistics
    total_employees = Employee.objects.count()
    today = timezone.now().date()
    today_attendance = Attendance.objects.filter(date=today)
    
    context = {
        'attendance_data': attendance_query,
        'total_employees': total_employees,
        'present_today': today_attendance.filter(status='present').count(),
        'absent_today': today_attendance.filter(status='absent').count(),
        'late_today': today_attendance.filter(status='late').count()
    }
    
    return render(request, 'report.html', context)

@login_required
def export_report(request):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="attendance_report_{start_date}_to_{end_date}.csv"'
    
    writer = csv.writer(response)
    writer.writerow(['Date', 'Staff Name', 'Status', 'In Time', 'Out Time'])
    
    # Add dummy data - replace with actual database query
    writer.writerow(['2024-03-01', 'John Doe', 'Present', '09:00 AM', '05:00 PM'])
    
    return response