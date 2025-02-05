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
    present_count = today_attendance.count()
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

# @login_required
# @user_passes_test(is_superuser)
# def settings_view(request):
#     users = User.objects.all().exclude(username=request.user.username)
#     context = {
#         'users': users,
#         'total_users': users.count(),
#         'active_users': users.filter(is_active=True).count(),
#         'staff_users': users.filter(is_staff=True).count(),
#         'superusers': users.filter(is_superuser=True).count()
#     }
#     return render(request, 'admin_panel.html', context)


# @login_required
# @user_passes_test(is_superuser)
# def update_user(request, user_id):
#     if request.method == 'POST':
#         user = User.objects.get(id=user_id)
#         user.is_active = request.POST.get('is_active') == 'true'
#         user.is_staff = request.POST.get('is_staff') == 'true'
#         user.is_superuser = request.POST.get('is_superuser') == 'true'
#         user.save()
#         messages.success(request, f'User {user.username} updated successfully')
#     return redirect('settings')


@login_required
def report_view(request):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    
    # Query attendance data
    attendance_query = Attendance.objects.select_related('emp').all()
    
    if start_date and end_date:
        attendance_query = attendance_query.filter(
            date__range=[start_date, end_date]
        )
    
    # Get statistics
    total_employees = Employee.objects.count()
    today = timezone.now().date()
    today_attendance = Attendance.objects.filter(date=today)
    
    # Calculate present/absent based on time_list
    present_count = today_attendance.exclude(time_list__isnull=True).exclude(time_list__exact='').count()
    absent_count = total_employees - present_count
    
    # Process attendance records
    processed_attendance = []
    for record in attendance_query:
        times = record.time_list.split(';') if record.time_list else []
        in_time = times[0] if times else '--:--'
        out_time = times[-1] if len(times) > 1 else '--:--'
        
        # Calculate working hours if both in and out times exist
        working_hours = 0
        if len(times) > 1:
            try:
                time_in = datetime.strptime(times[0], '%H:%M')
                time_out = datetime.strptime(times[-1], '%H:%M')
                diff = time_out - time_in
                working_hours = round(diff.total_seconds() / 3600, 2)
            except ValueError:
                working_hours = 0
        
        status = 'Absent'
        if times:
            status = 'Present' if len(times) > 1 else 'Present (No Out Time)'
        
        processed_attendance.append({
            'date': record.date,
            'emp_id': record.emp.emp_id,
            'emp_name': record.emp.emp_name,
            'department': record.emp.department,
            'in_time': in_time,
            'out_time': out_time,
            'working_hours': working_hours,
            'status': status
        })
    
    context = {
        'attendance_data': processed_attendance,
        'total_employees': total_employees,
        'present_today': present_count,
        'absent_today': absent_count
    }
    
    return render(request, 'report.html', context)

@login_required
def export_report(request):
    try:
        # Get filter parameters
        start_date = request.GET.get('start_date')
        end_date = request.GET.get('end_date')
        
        # Query attendance data
        attendance_query = Attendance.objects.select_related('emp').all()
        
        # Apply date filters if provided
        if start_date and end_date:
            attendance_query = attendance_query.filter(
                date__range=[start_date, end_date]
            )
        
        # Prepare response
        response = HttpResponse(content_type='text/csv')
        filename = "attendance_report"
        if start_date and end_date:
            filename += f"_{start_date}_to_{end_date}"
        response['Content-Disposition'] = f'attachment; filename="{filename}.csv"'
        
        # Write CSV data
        writer = csv.writer(response)
        writer.writerow([
            'Date', 
            'Employee ID', 
            'Employee Name', 
            'Department',
            'In Time', 
            'Out Time', 
            'Status'
        ])
        
        # Add attendance records
        for record in attendance_query:
            writer.writerow([
                record.date.strftime('%Y-%m-%d'),
                record.emp.emp_id,
                record.emp.emp_name,
                record.emp.department,
                record.time_in.strftime('%I:%M %p') if record.time_in else '--:--',
                record.time_out.strftime('%I:%M %p') if record.time_out else '--:--',
                record.status.title()
            ])
        
        return response
        
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        messages.error(request, 'Failed to export report')
        return redirect('report')
    

@login_required
@user_passes_test(is_superuser)
def manage_employees(request):
    employees = Employee.objects.all()
    
    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'add':
            emp_id = request.POST.get('emp_id')
            emp_name = request.POST.get('emp_name')
            department = request.POST.get('department')
            
            Employee.objects.create(
                emp_id=emp_id,
                emp_name=emp_name,
                department=department
            )
            messages.success(request, 'Employee added successfully')
            
        elif action == 'delete':
            emp_id = request.POST.get('emp_id')
            Employee.objects.filter(emp_id=emp_id).delete()
            messages.success(request, 'Employee deleted successfully')
            
        elif action == 'update':
            emp_id = request.POST.get('emp_id')
            emp_name = request.POST.get('emp_name')
            department = request.POST.get('department')
            
            Employee.objects.filter(emp_id=emp_id).update(
                emp_name=emp_name,
                department=department
            )
            messages.success(request, 'Employee updated successfully')
            
    context = {
        'employees': employees
    }
    return render(request, 'manage_employees.html', context)