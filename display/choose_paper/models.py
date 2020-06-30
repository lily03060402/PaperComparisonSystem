from django.db import models

# Create your models here.
"""
创建学生信息表模型
"""


"""
 该类是用来生成数据库的 必须要继承models.Model
"""
class Paper(models.Model):
    """
    创建如下几个表的字段
    """
    # 论文id primary_key=True: 该字段为主键
    articals = models.CharField('id', primary_key=True, max_length=15)
    # 论文作者 
    authors = models.TextField('作者')
    # 论文摘要 
    abstracts= models.TextField('摘要')
    # 论文作者单位
    affiliations= models.TextField('作者单位')
    # 论文标签
    IndexTerms= models.TextField('标签')
    # 论文关键字
    keywords= models.TextField('关键字')
    # 论文题目
    nodes= models.TextField('题目')
    # 论文引用关系，前面引用后面的
    inlinks= models.TextField('inlinks')
    # 论文引用关系，前面引用后面的
    outlinks= models.TextField('outlinks')
    GeneralTerms=models.TextField('GeneralTerms')
    # 指定表名 不指定默认APP名字——类名(app_demo_Student)
    class Meta:
        db_table = 'paper'

