from django.urls import path
from . import views

app_name = 'brainviewer'

urlpatterns = [
    path('', views.home_view, name='home'),
    path('<str:biosample_id>/<str:slice_number_str>/',
         views.viewer_view, name='viewer'),
    path('split/<str:biosample_id>/<str:slice_number_str>/',
         views.viewer_view_split, name='viewer'),
    path('split/test/<str:biosample_id>/<str:slice_number_str>/',
         views.viewer_view_split_test, name='viewer'),
]
