from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User
from django.contrib import messages
from django.http import HttpResponse
from django.utils import timezone
from .models import Employee, Attendance
from datetime import datetime, date
import logging
import csv
from django.contrib import messages
from django.conf import settings
import os
import requests
import shutil

logger = logging.getLogger(__name__)

def is_superuser(user):
    return user.is_superuser

@login_required
def home_view(request):
    today = date.today()
    
    # Get today's attendance
    today_attendance = Attendance.objects.select_related('emp').filter(date=today)
    
    # Calculate present/absent counts
    total_employees = Employee.objects.count()
    present_count = today_attendance.count()
    absent_count = total_employees - present_count
    
    # Format attendance data
    formatted_attendance = [{
        'emp_id': att.emp.emp_id,
        'emp_name': att.emp.emp_name,
        'department': att.emp.department,
        'in_times': att.get_in_time_array(),
        'out_times': att.get_out_time_array(),
        'working_hours': att.calculate_working_hours(),
        'status': att.get_status()
    } for att in today_attendance]
    
    context = {
        'present_count': present_count,
        'absent_count': absent_count,
        'attendance_data': formatted_attendance,
        'current_date': today
    }
    
    return render(request, 'home.html', context)

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
    
    # Format report data
    report_data = [{
        'date': att.date,
        'emp_id': att.emp.emp_id,
        'emp_name': att.emp.emp_name,
        'department': att.emp.department,
        'in_times': att.get_in_time_array(),
        'out_times': att.get_out_time_array(),
        'working_hours': att.calculate_working_hours(),
        'status': att.get_status()
    } for att in attendance_query]
    
    # Get statistics
    total_employees = Employee.objects.count()
    present_employees = len(set(att['emp_id'] for att in report_data))
    
    context = {
        'attendance_data': report_data,
        'total_employees': total_employees,
        'present_employees': present_employees,
        'start_date': start_date,
        'end_date': end_date
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
    

def store_embeddings(db_path, output_file="backend/face_embeddings.json"):
    url = "http://127.0.0.1:7000/store_embeddings/"
    payload = {"db_path": db_path, "output_file": output_file}
    response = requests.post(url, json=payload)  # Sending as JSON body
    return response.json()


def load_embeddings(output_file="backend/face_embeddings.json"):
    url = "http://127.0.0.1:7000/load_embeddings/"
    params = {"input_file": output_file}
    response = requests.get(url, params=params)
    return response.json()

@login_required
@user_passes_test(is_superuser)
def manage_employees(request):
    employees = Employee.objects.all()
    
    if request.method == 'POST':
        action = request.POST.get('action')
        
        if action == 'add':
            try:
                emp_id = request.POST.get('emp_id')
                emp_name = request.POST.get('emp_name')
                department = request.POST.get('department')
                
                # Create employee
                Employee.objects.create(
                    emp_id=emp_id,
                    emp_name=emp_name,
                    department=department
                )
                
                # Handle image uploads
                if request.FILES.getlist('images'):
                    # Create temporary directory for employee
                    temp_dir = os.path.join(settings.MEDIA_ROOT, emp_id)
                    # temp_dir = os.path.join('../media/temp/', emp_id)
                    print(temp_dir)
                    os.makedirs(temp_dir, exist_ok=True)
                    # Save images to temporary directory
                    for image in request.FILES.getlist('images'):
                        image_path = os.path.join(temp_dir, image.name)
                        with open(image_path, 'wb+') as destination:
                            for chunk in image.chunks():
                                destination.write(chunk)
                    print('Image written successfully!')
                    # Call store_embeddings API
                    try:
                        embeddings_file = r'../backend/face_embeddings.json'
                        print("before api call")

                        response = store_embeddings(settings.MEDIA_ROOT, embeddings_file)
                        print(response)
                        if response.get('status') == 'success':
                            print('Success')
                            messages.success(request, 'Employee added and embeddings stored successfully')
                        else:
                            messages.warning(request, 'Employee added but embeddings storage failed')
                        
                        # Clean up temporary directory after processing
                        shutil.rmtree(temp_dir)
                        
                    except Exception as e:
                        messages.error(request, f'Error processing embeddings: {str(e)}')
                        # Keep temp directory in case of error for debugging
                
                messages.success(request, 'Employee added successfully')
                
            except Exception as e:
                messages.error(request, f'Failed to add employee: {str(e)}')
        
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

