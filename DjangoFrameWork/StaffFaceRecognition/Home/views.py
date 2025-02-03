from django.shortcuts import render
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User
import logging

logger = logging.getLogger(__name__)

def is_superuser(user):
    return user.is_superuser

@login_required
def home_view(request):
    return render(request, 'home.html', {
        'user': request.user,
        'is_superuser': request.user.is_superuser
    })

@login_required
@user_passes_test(is_superuser)
def settings_view(request):
    users = User.objects.all().exclude(username=request.user.username)
    logger.info(f"Found {users.count()} users")
    return render(request, 'settings.html', {'users': users})