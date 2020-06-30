from django.urls import path

from . import views


app_name = 'choose_paper'
urlpatterns = [
    path('insert/',views.insert, name='insert'),
    path('index/',views.index,name='index'),
    path('paper_detail/<int:paper_id>/',
         views.paper_detail, name='paper_detail'),
    path('choose_detail/', views.choose_detail, name='choose_detail'),
    path('setpaper/', views.function, name='setPaper'),

    # path('/find/',views.find, name='find'),
    # path('/delete/<int:studentNum>/',views.delete, name='delete'),

]