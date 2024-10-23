from django.urls import path

from . import views

urlpatterns = [
    path('', views.login_index, name="login_index"),
    path('index', views.index, name="index"),
    path('logout', views.logout, name="logout"),
    path('login', views.login, name="login"),
    path('eval', views.new_evaluation, name='new_evaluation'),
    path('tutorial', views.tutorial, name='tutorial'),
]