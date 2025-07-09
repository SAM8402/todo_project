"""
URL configuration for todo_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from rest_framework.authtoken.views import obtain_auth_token
from django.conf.urls.static import static
from django.conf import settings
from django.views.generic import RedirectView
from django.views.static import serve
from pathlib import Path


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', RedirectView.as_view(url='/viewer/', permanent=False)),
    path('api-auth/', include('rest_framework.urls')),
    # path('accounts/', include('django.contrib.auth.urls')),
    # path('api-token-auth/', obtain_auth_token),
    path('viewer/', include('brainviewer.urls')),
    path('images/<path:path>', serve,
         {'document_root': settings.BASE_DIR / 'brainviewer/static/images_data'}),

] 
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
urlpatterns += static(settings.STATIC_URL,
                      document_root=settings.STATICFILES_DIRS[0])
