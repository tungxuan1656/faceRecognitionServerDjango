from django.urls import path
from . import views

urlpatterns = [
    path('a', views.index),
    path('a/<str:s>', views.index2),
]
