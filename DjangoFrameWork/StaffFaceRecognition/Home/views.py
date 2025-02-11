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
        'time_in_list': att.get_in_time()[0] if att.get_in_time() else '--:--',
        'time_out_list': att.get_out_time()[-1] if att.get_out_time() else '--:--',
        'working_hours': att.get_working_hours(),
    } for att in today_attendance]
    
    context = {
        'present_count': present_count,
        'attendance_data': formatted_attendance,
        'current_date': today,
        'total_staff':total_employees,
    }
    print(context)
    return render(request, 'home.html', context)

@login_required
def report_view(request):
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    department = request.GET.get('department')
    
    # Query attendance data
    attendance_query = Attendance.objects.select_related('emp').all()
    
    if start_date and end_date:
        attendance_query = attendance_query.filter(
            date__range=[start_date, end_date]
        )
    
    if department:
        attendance_query = attendance_query.filter(
            emp__department=department
        )
    # Get unique departments for filter dropdown
    departments = Employee.objects.values_list('department', flat=True).distinct()
    
    # Format report data
    attendance_data = [{
        'date': att.date,
        'emp_id': att.emp.emp_id,
        'emp_name': att.emp.emp_name,
        'department': att.emp.department,
        'time_in_list': att.get_in_time()[0] if att.get_in_time() else '--:--',
        'time_out_list': att.get_out_time()[-1] if att.get_out_time() else '--:--',
        'working_hours': att.get_working_hours(),
    } for att in attendance_query]
    
    context = {
        'attendance_data': attendance_data,
        'departments': departments,
        'start_date': start_date,
        'end_date': end_date,
        'department': department
    }
    for k,v in context.items(): print(k, v)
    return render(request, 'report.html', context)

@login_required
def export_report(request):
    try:
        # Get filter parameters
        start_date = request.GET.get('start_date')
        end_date = request.GET.get('end_date')
        department = request.GET.get('department')  # Added department filter to match report_view
        
        # Query attendance data
        attendance_query = Attendance.objects.select_related('emp').all()
        
        # Apply filters if provided
        if start_date and end_date:
            try:
                # Convert string dates to datetime objects
                start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
                end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
                attendance_query = attendance_query.filter(date__range=[start_date, end_date])
            except ValueError:
                messages.error(request, 'Invalid date format. Use YYYY-MM-DD')
                return redirect('report')

        if department:
            attendance_query = attendance_query.filter(emp__department=department)
        
        # Check if any records exist
        if not attendance_query.exists():
            messages.warning(request, 'No attendance records found for the selected period')
            return redirect('report')
        
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
            'Working Hours'
        ])
        
        # Add attendance records
        for record in attendance_query:
            in_time = record.get_in_time()[0] if record.get_in_time() else '--:--'
            out_time = record.get_out_time()[-1] if record.get_out_time() else '--:--'
            
            writer.writerow([
                record.date.strftime('%Y-%m-%d'),
                record.emp.emp_id,
                record.emp.emp_name,
                record.emp.department,
                in_time,
                out_time,
                record.get_working_hours(),
            ])
        
        return response
        
    except Exception as e:
        logger.error(f"Export failed: {str(e)}", exc_info=True)
        messages.error(request, 'Failed to export report')
        return redirect('report')

def store_embeddings(db_path, output_file="backend/face_embeddings.json"):
    url = "http://127.0.0.1:5600/store_embeddings/"
    payload = {"db_path": db_path, "output_file": output_file}
    response = requests.post(url, json=payload)  # Sending as JSON body
    return response.json()


def load_embeddings(output_file="backend/face_embeddings.json"):
    url = "http://127.0.0.1:5600/load_embeddings/"
    params = {"input_file": output_file}
    response = requests.get(url, params=params)
    return response.json()

@login_required
@user_passes_test(is_superuser)
def manage_employees(request):
    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'add':
            try:
                emp_id = request.POST.get('emp_id')
                emp_name = request.POST.get('emp_name')
                department = request.POST.get('department')
                
                # Check if employee already exists (optional)
                if Employee.objects.filter(emp_id=emp_id).exists():
                    messages.error(request, f'Employee with ID {emp_id} already exists.')
                else:
                    # Create employee record
                    Employee.objects.create(
                        emp_id=emp_id,
                        emp_name=emp_name,
                        department=department
                    )
                    
                    # Handle image uploads
                    images = request.FILES.getlist('images')
                    if images:
                        # Create a temporary directory for this employee's images
                        temp_dir = os.path.join(settings.MEDIA_ROOT, emp_id)
                        os.makedirs(temp_dir, exist_ok=True)
                        for image in images:
                            image_path = os.path.join(temp_dir, image.name)
                            with open(image_path, 'wb+') as destination:
                                for chunk in image.chunks():
                                    destination.write(chunk)
                        print(f'Images saved to: {temp_dir}')
                        
                        # Call the store_embeddings API (adjust paths as needed)
                        try:
                            embeddings_file = os.path.join(settings.BASE_DIR, 'backend', 'face_embeddings.json')
                            print("Before API call to store embeddings")
                            
                            response = store_embeddings(settings.MEDIA_ROOT, embeddings_file)
                            print("Embeddings API response:", response)
                            
                            if response.get('status') == 'success':
                                messages.success(request, 'Employee added and embeddings stored successfully.')
                            else:
                                messages.warning(request, 'Employee added but embeddings storage failed.')
                            
                            # Clean up temporary directory after processing
                            shutil.rmtree(temp_dir)
                        except Exception as e:
                            messages.error(request, f'Error processing embeddings: {str(e)}')
                    else:
                        messages.success(request, 'Employee added successfully.')
            except Exception as e:
                messages.error(request, f'Failed to add employee: {str(e)}')
                
        elif action == 'delete':
            try:
                emp_id = request.POST.get('emp_id')
                Employee.objects.filter(emp_id=emp_id).delete()
                messages.success(request, 'Employee deleted successfully.')
            except Exception as e:
                messages.error(request, f'Failed to delete employee: {str(e)}')
        
        # Redirect to avoid resubmission on refresh.
        return redirect('manage_employees')
    
    employees = Employee.objects.all()
    context = {
        'employees': employees
    }
    return render(request, 'manage_employees.html', context)

