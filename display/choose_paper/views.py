from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
from choose_paper.models import Paper
import random
from pyecharts.charts import Bar
from pyecharts import options as opts
from pyecharts.charts import Graph
from pyecharts.charts import Line
from pyecharts.charts import Pie
from pyecharts.charts import WordCloud
from pyecharts.globals import ThemeType
from choose_paper.PaperCompare.prediction import predmodel,predmodel_norules
import os
import numpy as np
import json


# 插入数据到数据库中
def insert(request):
    with open('choose_paper/data/articles.txt', 'r', encoding='utf-8') as f1:
        with open('choose_paper/data/abstracts.txt', 'r', encoding='utf-8') as f2:
            with open('choose_paper/data/authors.txt', 'r', encoding='utf-8') as f3:
                with open('choose_paper/data/affiliations.txt', 'r', encoding='utf-8') as f4:
                    with open('choose_paper/data/GeneralTerms.txt', 'r', encoding='utf-8') as f5:
                        with open('choose_paper/data/indexTerms.txt', 'r', encoding='utf-8') as f6:
                            with open('choose_paper/data/inlinks_nodes.txt', 'r', encoding='utf-8') as f7:
                                with open('choose_paper/data/keywords.txt', 'r', encoding='utf-8') as f8:
                                    with open('choose_paper/data/nodes.txt', 'r', encoding='utf-8') as f9:
                                        with open('choose_paper/data/outlinks_nodes.txt', 'r', encoding='utf-8') as f10:
                                            articles = f1.readlines()
                                            abstracts = f2.readlines()
                                            authors = f3.readlines()
                                            affiliations = f4.readlines()
                                            GeneralTerms = f5.readlines()
                                            indexTerms = f6.readlines()
                                            inlinks = f7.readlines()
                                            keywords = f8.readlines()
                                            nodes = f9.readlines()
                                            outlinks = f10.readlines()
                                            for i in range(20):
                                                paper = Paper()
                                                paper.articals = articles[i].strip(
                                                )
                                                paper.abstracts = abstracts[i]
                                                paper.affiliations = affiliations[i]
                                                paper.authors = authors[i]
                                                paper.GeneralTerms = GeneralTerms[i]
                                                paper.inlinks = inlinks[i]
                                                paper.IndexTerms = indexTerms[i]
                                                paper.keywords = keywords[i]
                                                paper.nodes = nodes[i]
                                                paper.outlinks = outlinks[i]

                                                paper.save()

    return HttpResponse('数据插入完毕')

# 实现初始界面的跳转等操作
def index(request):  # 返回论文集合的论文标题
    paper_list = Paper.objects.all()

# 不同年份上论文集的个数（柱状图）-------------------------------------------
# 读取years.txt，将年份对应的论文数放入一个字典
    file_years = 'D:/code/django/display/choose_paper/data/years.txt'
    fyears = open(file_years, 'r')
    years_dic = {}
    for line in fyears:
        year = line.strip()
        if year in years_dic.keys() and year is not '':
            years_dic[year] += 1
        elif year is not '':
            temp = {}
            temp[year] = 1
            years_dic.update(temp)
        else:
            continue
    del years_dic['994']
    fyears.close()
# 构造x,y轴的数据
    x_data = []
    y_data = []
    for k in sorted(years_dic):
        x_data.append(k)
        y_data.append(years_dic[k])

    year_bar = Bar(init_opts=opts.InitOpts(width='1200px'))\
        .add_xaxis(xaxis_data=x_data)\
        .add_yaxis(series_name="论文数量", y_axis=y_data)
    year_bar.set_global_opts(title_opts=opts.TitleOpts(title='数据集中不同年份的论文数量'),
                             visualmap_opts=opts.VisualMapOpts(
        is_show=True,
        type_='color',
        is_piecewise=True,
        pieces=[
            {"min": 3000, "color": '#F01C06'},
            {"min": 2000, "max": 2999, "color": '#F93F2B'},
            {"min": 1000, "max": 1999, "color": '#FB8275'},
            {"min": 1000, "max": 1999, "color": '#FAA62A'},
            {"min": 500, "max": 999, "color": '#F4E362'},
            {"min": 100, "max": 499, "color": '#45F172'},
            {"min": 50, "max": 99, "color": '#7FC3F1'},
            {"max": 49, "color": '#C4E3F8'}
        ],
        orient='vertical'
    ))
    year_bar.render(path="choose_paper/data/graph/graph_year_num.txt")  # 显示图表
    graph_year_num_txt = "{"
    with open('choose_paper/data/graph/graph_year_num.txt', 'r', encoding='utf-8') as f:
        line = f.readlines()
        for i in range(15, len(line)-4):
            graph_year_num_txt += line[i]
#-------------------------------------------------------------------------------------
# 数据预处理
# 读取years.txt，将年份对应的论文数放入一个字典
    file_years = 'choose_paper/data/years.txt'
    fyears = open(file_years, 'r')
    years_dic = {}
    for line in fyears:
        year = line.strip()
        if year in years_dic.keys() and year is not '':
            years_dic[year] += 1
        elif year is not '':
            temp = {}
            temp[year] = 1
            years_dic.update(temp)
        else:
            continue
    del years_dic['994']
# 获取每篇论文的引用量
    ilinkspath = 'choose_paper/data/quotes.txt'
    ifile = open(ilinkspath, 'r')
    innum = []

    for line in ifile:
        num = line.strip('\n')
        innum.append(num)
    ifile.close()

# 获取不同年份论文的总被引用量
    file_years = 'choose_paper/data/years.txt'
    fyears = open(file_years, 'r')
    innum_dic = {}
# 论文索引
    index = 0
    for line in fyears:
        year = line.strip()
        if year in innum_dic.keys() and year is not '':
            if innum[index] is not ' ' and innum[index] is not '':
                innum_dic[year] += int(innum[index])
        elif year is not '':
            temp = {}
            if innum[index] is ' ' or innum[index] is '':
                temp[year] = 0

            else:
                temp[year] = int(innum[index])
            innum_dic.update(temp)
        else:
            index += 1
            continue
        index += 1

    del innum_dic['994']
    fyears.close()

# 获取不同年份论文的平均引用量
    inavg_dic = {}
    for k in innum_dic:
        temp = {}
        temp[k] = int(innum_dic[k]) // int(years_dic[k])
    # print(num_dic[k],years_dic[k],temp[k])
        inavg_dic[k] = temp[k]

# 构造x,y轴的数据
    x_data = []
    y_data = []
    for k in sorted(inavg_dic):
        x_data.append(k)
        y_data.append(inavg_dic[k])

# 可视化
# 可绘制折线图
    line = Line()\
        .add_xaxis(xaxis_data=x_data)\
        .add_yaxis(
        series_name="平均被引用量",
        y_axis=y_data,
        markpoint_opts=opts.MarkPointOpts(
            data=[
                opts.MarkPointItem(type_="max", name="最大值"),
                opts.MarkPointItem(type_="min", name="最小值"),
            ]
        ),
        markline_opts=opts.MarkLineOpts(
            data=[opts.MarkLineItem(type_="average", name="平均值")]
        ),
    )\
    .set_global_opts(
        title_opts=opts.TitleOpts(title="不同年份平均被引用量统计", subtitle="统计截止到2020.6.26"),
        xaxis_opts=opts.AxisOpts(type_="category", boundary_gap=False),
        datazoom_opts=opts.DataZoomOpts(is_show= True,orient="horizontal")
    )
    line.render(path="choose_paper/data/graph/line.txt")  # 显示图表
    line_txt = "{"
    with open('choose_paper/data/graph/line.txt', 'r', encoding='utf-8') as f:
        line = f.readlines()
        for i in range(15, len(line)-4):
            line_txt += line[i]
# 不同年份上论文集的平均引用量、被引用量（柱状图）------------------------------------------
    # 数据预处理：获取每篇论文的引用量
    # -----------------outlinks----------------------------
    olinkspath = 'D:/code/django/display/choose_paper/data/outlinks.txt'
    ofile = open(olinkspath, 'r')
    outnum = []

    for line in ofile:
        lines = line.strip('\n').split()
        num = len(lines)
        outnum.append(num)
    ofile.close()
# 数据预处理：获取不同年份论文的总引用量
    file_years = 'D:/code/django/display/choose_paper/data/years.txt'
    fyears = open(file_years, 'r')
    num_dic = {}
# 论文索引
    index = 0

    for line in fyears:
        year = line.strip()
        if year in num_dic.keys() and year is not '':
            num_dic[year] += outnum[index]
        elif year is not '':
            temp = {}
            temp[year] = outnum[index]
            num_dic.update(temp)
        else:
            index += 1
            continue
        index += 1

    del num_dic['994']
    fyears.close()
    # 构造x,y轴的数据
    x_data1 = []
    y_data1 = []
    for k in sorted(num_dic):
        x_data1.append(k)
        y_data1.append(num_dic[k])

# ----------------------------inlinks--------------------------------
    olinkspath = 'D:/code/django/display/choose_paper/data/inlinks.txt'
    ofile = open(olinkspath, 'r')
    outnum = []

    for line in ofile:
        lines = line.strip('\n').split()
        num = len(lines)
        outnum.append(num)
    ofile.close()
# 数据预处理：获取不同年份论文的总引用量
    file_years = 'D:/code/django/display/choose_paper/data/years.txt'
    fyears = open(file_years, 'r')
    num_dic = {}
# 论文索引
    index = 0

    for line in fyears:
        year = line.strip()
        if year in num_dic.keys() and year is not '':
            num_dic[year] += outnum[index]
        elif year is not '':
            temp = {}
            temp[year] = outnum[index]
            num_dic.update(temp)
        else:
            index += 1
            continue
        index += 1

    del num_dic['994']
    fyears.close()
    # 构造x,y轴的数据
    x_data2 = []
    y_data2 = []
    for k in sorted(num_dic):
        x_data2.append(k)
        y_data2.append(num_dic[k])

    inavg_bar = Bar(init_opts=opts.InitOpts(width='1200px'))\
        .add_xaxis(xaxis_data=x_data2)\
        .add_yaxis(series_name="平均被引用量", y_axis=y_data2)\
        .add_yaxis(series_name="平均引用量", y_axis=y_data1)
    inavg_bar.set_global_opts(title_opts=opts.TitleOpts(title='不同年份论文的平均被引用量与引用量'),
                              datazoom_opts=opts.DataZoomOpts(is_show=True,
                                                              orient="horizontal"))
    inavg_bar.render(path="choose_paper/data/graph/graph_avg.txt")  # 显示图表
    graph_avg_txt = "{"
    with open('choose_paper/data/graph/graph_avg.txt', 'r', encoding='utf-8') as f:
        line = f.readlines()
        for i in range(15, len(line)-4):
            graph_avg_txt += line[i]
#---------------------------------------------------------------------------------------
# 不同会议的平均被引用数统计
# 读取venues.txt，将不同会议对应的论文数放入一个字典

    file_venues = 'choose_paper/data/venues.txt'
    fvenues = open(file_venues,'r')
    venues_dic = { }
    for line in fvenues:
        venue = line.strip()
        if venue in venues_dic.keys() and venue is not '':
            venues_dic[venue]+=1
        elif venue is not '':
            temp = {}
            temp[venue] = 1
            venues_dic.update(temp)
        else:
            continue
    fvenues.close()

    f= open(file_venues,'r')
# 获取不同会议论文的总被引用量
    quote_dic = {}
    index = 0
    for line in f:
        venue = line.strip()
        if venue in quote_dic.keys() and venue is not '':
            if innum[index] is not ' 'and innum[index] is not '':
                quote_dic[venue]+=int(innum[index])
        elif venue is not '':
            temp = {}
            if innum[index] is ' 'or  innum[index] is '':
                temp[venue] =0
            else:
                temp[venue] = int(innum[index])
            quote_dic.update(temp)
        else:
            index+=1
            continue
        index +=1

# 数据预处理：获取不同会议论文的平均引用量
    avg_dic = {}
    for k in quote_dic:
        temp = {}
        temp[k] = round(quote_dic[k] / venues_dic[k], 2)
    # print(num_dic[k],years_dic[k],temp[k])
        avg_dic[k] = temp[k]

# 对数据进行筛选，选出排名前60的会议
    data_list = [{k: v} for k, v in avg_dic.items()]
    f = lambda x: list(x.values())[0]
    l=sorted(data_list, key=f, reverse=True)
    filter_l = l[:60]
# 构造x,y数据
    x_data1 = []
    y_data1 = []
    for item in filter_l:
        for k in item:
            x_data1.append(k)
            y_data1.append(item[k])

#绘制饼图
    venue_pie=Pie()\
    .add("会议/期刊", [list(z) for z in zip(x_data1, y_data1)],
        label_opts=opts.LabelOpts(is_show=False),
        radius=[40, 120])\
    .set_colors(["blue", "green", "yellow", "red", "pink", "orange", "purple","grey"])\
    .set_global_opts(title_opts=opts.TitleOpts(
                               title="不同会议/期刊的论文平均被引用量（排名前60）",),
                    legend_opts=opts.LegendOpts(is_show=False))\
    .set_series_opts(tooltip_opts=opts.TooltipOpts(
             formatter="{a} <br/>{b}: {c} ({d}%)"
        ),)

    venue_pie.render(path="choose_paper/data/graph/venue_pie.txt")  # 显示图表
    venue_pie_txt = "{"
    with open('choose_paper/data/graph/venue_pie.txt', 'r', encoding='utf-8') as f:
        line = f.readlines()
        for i in range(15, len(line)-4):
            venue_pie_txt += line[i]


    context = {
        'paper_list': paper_list,
        'graph_year_num_txt': graph_year_num_txt,
        'graph_avg_txt': graph_avg_txt,
        'line_txt':line_txt,
        'venue_pie_txt':venue_pie_txt
    }
    return render(request, 'choose_paper/index.html', context)

# 显示一篇论文的详细内容(存到数据库中的/侧边栏)
def paper_detail(request, paper_id):  
    paper = Paper.objects.get(articals=paper_id)
    context = {
        'articals': paper.articals,
        'authors ': paper.authors,
        'abstracts': paper.abstracts,
        'affiliations': paper.affiliations,
        'IndexTerms ': paper.IndexTerms,
        'keywords': paper.keywords,
        'nodes': paper.nodes,
        'inlinks': paper.inlinks,
        'outlinks': paper.outlinks,
        'GeneralTerms': paper.GeneralTerms,

    }
    return render(request, 'choose_paper/paper_example.html', context)


 # 根据会议和年份筛选数据集合
def function(request): 
    if request.POST:
        # 详情 按钮响应事件
        if 'paper_set' in request.POST:
            year = request.POST['datetimepicker']
            venue = request.POST['venue']
            articals = []
            with open("choose_paper/data/id_year_venue.txt", 'r', encoding='utf-8') as f:
                lines = f.readlines()
                i = 0
                while i < len(lines)-3:
                    a = lines[i].strip()
                    b = lines[i+1].strip()
                    c = lines[i+2].strip()
                    if year == "":
                        if(c == venue):
                            articals.append(a)
                        i += 4
                    elif venue == "":
                        if(b == year):
                            articals.append(a)
                        i += 4

                    else:
                        if(b == year and c == venue):
                            articals.append(a)
                        i += 4
            
            name=[]
            #找论文名称
            with open('choose_paper/data/nodes2.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i in range(len(articals)):
                    for j in range(len(lines)):
                       line=lines[j].split()
                       if(line[0]==articals[i]):
                           s=""
                           for m in range(1,len(line)):
                               s=s+line[m]
                               s=s+" "
                           name.append(s)
                           break
            ##词云--------------------------------------
            # 年份列表
            file_years = 'choose_paper/data/years.txt'
            years = []
            with open(file_years,'r',encoding="utf-8") as f:
             for line in f.readlines():
                year_ = line.strip()
                years.append(year_)
            # 会议列表
            file_venues = 'choose_paper/data/venues.txt'
            venues = []
            with open(file_venues,'r',encoding="utf-8") as f:
                for line in f.readlines():
                    venue_ = line.strip()
                    venues.append(venue_)
            # 去除停用词的关键词列表
            stopword = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
                'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
                'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
                'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
                'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
                'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd',
                'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven',
                'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn'])

            # 关键词列表
            key_path = 'choose_paper/data/keywords.txt'
            key_file = open(key_path,'r')
            word_list =[]
            index = 0
            for line in key_file:
                words = line.strip().split()
                for word in words:
                    if word in stopword:
                     words.remove(word)
                word_list.append(words)
            # 定义根据年份会议获取数据
            word_dict = {} #某年某个关键词出现的次数统计
            use_data = [] #可视化时传入的data
            yea = str(year) #将输入的年份转为str
            venu = str(venue)
            for i in range(len(years)):
                if yea == years[i].strip() and venu == venues[i].strip():
                    sword = word_list[i]
                    for word in sword:
                        if word in word_dict.keys() and word is not '':
                            word_dict[word]+=1
                        elif word is not '':
                            temp = { }
                            temp[word]=1
                            word_dict.update(temp)
                        else:
                             continue
             # 对word_dict进行处理，筛选词语
            length = len(word_dict)
            if length>60:
                for k,v in word_dict.items():
                    if v>1:
                        tup = (k,v)
                        use_data.append(tup)
            else:
                for k,v in word_dict.items():
                    tup = (k,v)
                    use_data.append(tup)
            ##词云展示：
            wtitle  = str(year)+' '+venue+':关键词词云展示'
            mywordcloud = WordCloud()
            mywordcloud.add('',use_data,shape='triangle')\
               .set_global_opts(title_opts=opts.TitleOpts(
                               title=wtitle))
            mywordcloud.render(path="choose_paper/data/graph/wordCloud.txt")  # 显示图表
            mywordcloud_txt = "{"
            with open('choose_paper/data/graph/wordCloud.txt', 'r', encoding='utf-8') as f:
                line = f.readlines()
                for i in range(16, len(line)-4):
                    mywordcloud_txt += line[i]
              
           
            context = {
                'nodes': name , # 获取满足年份，会议的论文id
                'mywordcloud_txt':mywordcloud_txt,
                
            }

            return render(request, 'choose_paper/yv_paper.html', context)
# 分析 按钮响应事件
        else:
            year = request.POST['datetimepicker']
            venue = request.POST['venue']
            articals = []#获得论文编号
            with open("choose_paper/data/id_year_venue.txt", 'r', encoding='utf-8') as f:
                lines = f.readlines()
                i = 0
                while i < len(lines)-3:
                    a = lines[i].strip()
                    b = lines[i+1].strip()
                    c = lines[i+2].strip()
                    if year == "":
                        if(c == venue):
                            articals.append(int(a))
                        i += 4
                    elif venue == "":
                        if(b == year):
                            articals.append(int(a))
                        i += 4
                    else:
                        if(b == year and c == venue):
                            articals.append(int(a))
                        i += 4
            
            #进行训练
            dictAll0=np.load('choose_paper/PaperCompare/data_process_result/dictAll0.npy').item()
            dictAll1=np.load('choose_paper/PaperCompare/data_process_result/dictAll1.npy').item()
            dictAll2=np.load('choose_paper/PaperCompare/data_process_result/dictAll2.npy').item()
            dictAll3=np.load('choose_paper/PaperCompare/data_process_result/dictAll3.npy').item()
            dictAll4=np.load('choose_paper/PaperCompare/data_process_result/dictAll4.npy').item()
            dict0 = np.load('choose_paper/PaperCompare/data_process_result/dict0.npy').item()
            dict1 = np.load('choose_paper/PaperCompare/data_process_result/dict1.npy').item()
            dict2 = np.load('choose_paper/PaperCompare/data_process_result/dict2.npy').item()
            dict3 = np.load('choose_paper/PaperCompare/data_process_result/dict3.npy').item()  
            dict4 = np.load('choose_paper/PaperCompare/data_process_result/dict4.npy').item()
            #-------------------------------------------------model0------------------
            if('check' in request.POST):#有规则的
                finalresult=predmodel("choose_paper/PaperCompare/model/model0.h5",articals,dict0,dictAll0,0)
            else:   #无规则的
                finalresult=predmodel_norules("choose_paper/PaperCompare/model_/model0.h5",articals,dict0,dictAll0,0)
            
            result0=[]#记录最后的相似论文对的论文名字
            for tmp in finalresult:
                result_0=""
                for j in range(2):
                    with open("choose_paper/data/nodes2.txt",'r',encoding='utf-8') as f:
                      lines=f.readlines()
                      for i in range(len(lines)):
                          line=lines[i].split()
                          if(str(tmp[0][j])==str(line[0])):  
                              for m in range(1,len(line)):
                                  result_0+=line[m]
                                  result_0+=" "
                    if(j==0):
                      result_0+=","
                result0.append(result_0)
            other0=[]
            for tmp in finalresult:
                other0.append(tmp[1])
            #-------------------------------------------------model1------------------
            if('check' in request.POST):#有规则的
                finalresult=predmodel("choose_paper/PaperCompare/model/model1.h5",articals,dict1,dictAll1,1)
            else:#无规则的
                finalresult=predmodel_norules("choose_paper/PaperCompare/model_/model1.h5",articals,dict1,dictAll1,1)
            result1=[]#记录最后的相似论文对的论文名字
            
            for tmp in finalresult:
                result_1=""
                for j in range(2):
                    with open("choose_paper/data/nodes2.txt",'r',encoding='utf-8') as f:
                      lines=f.readlines()
                      for i in range(len(lines)):
                          line=lines[i].split()
                          if(str(tmp[0][j])==str(line[0])):  
                              for m in range(1,len(line)):
                                  result_1+=line[m]
                                  result_1+=" "
                    if(j==0):
                      result_1+=","
                # result_1+=")"
                result1.append(result_1)
            other1=[]
            for tmp in finalresult:
                other1.append(tmp[1])
            #-------------------------------------------------model2------------------
            if('check' in request.POST):#有规则的
                finalresult=predmodel("choose_paper/PaperCompare/model/model2.h5",articals,dict2,dictAll2,2)
            else:#无规则的
                finalresult=predmodel_norules("choose_paper/PaperCompare/model_/model2.h5",articals,dict2,dictAll2,2)
            result2=[]#记录最后的相似论文对的论文名字
            
            for tmp in finalresult:
                result_2=""
                for j in range(2):
                    with open("choose_paper/data/nodes2.txt",'r',encoding='utf-8') as f:
                      lines=f.readlines()
                      for i in range(len(lines)):
                          line=lines[i].split()
                          if(str(tmp[0][j])==str(line[0])):  
                              for m in range(1,len(line)):
                                  result_2+=line[m]
                                  result_2+=" "
                    if(j==0):
                      result_2+=","
                # result_2+=")"
                result2.append(result_2)
            other2=[]
            for tmp in finalresult:
                other2.append(tmp[1])
            #-------------------------------------------------model3------------------
            if('check' in request.POST):#有规则的
                finalresult=predmodel("choose_paper/PaperCompare/model/model3.h5",articals,dict3,dictAll3,3)
            else:#无规则的
                finalresult=predmodel_norules("choose_paper/PaperCompare/model_/model3.h5",articals,dict3,dictAll3,3)
            result3=[]#记录最后的相似论文对的论文名字
            
            for tmp in finalresult:
                result_3=""
                for j in range(2):
                    with open("choose_paper/data/nodes2.txt",'r',encoding='utf-8') as f:
                      lines=f.readlines()
                      for i in range(len(lines)):
                          line=lines[i].split()
                          if(str(tmp[0][j])==str(line[0])):  
                              for m in range(1,len(line)):
                                  result_3+=line[m]
                                  result_3+=" "
                    if(j==0):
                      result_3+=","
                # result_3+=")"
                result3.append(result_3)
            other3=[]
            for tmp in finalresult:
                other3.append(tmp[1])
            #-------------------------------------------------model4------------------
            if('check' in request.POST):#有规则的
                finalresult=predmodel("choose_paper/PaperCompare/model/model4.h5",articals,dict4,dictAll4,4)
            else:#无规则的
                finalresult=predmodel_norules("choose_paper/PaperCompare/model_/model4.h5",articals,dict4,dictAll4,4)
            result4=[]#记录最后的相似论文对的论文名字
            
            for tmp in finalresult:
                result_4=""
                for j in range(2):
                    with open("choose_paper/data/nodes2.txt",'r',encoding='utf-8') as f:
                      lines=f.readlines()
                      for i in range(len(lines)):
                          line=lines[i].split()
                          if(str(tmp[0][j])==str(line[0])):  
                              for m in range(1,len(line)):
                                  result_4+=line[m]
                                  result_4+=" "
                    if(j==0):
                      result_4+=","
                # result_4+=")"
                result4.append(result_4)
            other4=[]
            for tmp in finalresult:
                other4.append(tmp[1])
            #通过json传递数据
            arr0 = json.dumps(result0)
            arr1= json.dumps(result1)
            arr2 = json.dumps(result2)
            arr3 = json.dumps(result3)
            arr4 = json.dumps(result4)
            otr0=json.dumps(other0)
            otr1=json.dumps(other0)
            otr2=json.dumps(other0)
            otr3=json.dumps(other0)
            otr4=json.dumps(other0)
            context = {
                'back': arr0,  # 获取满足年份，会议的论文id
                'issue': arr1,
                'contribute': arr2,
                'measure': arr3,
                'exam': arr4,
                'other0':otr0,
                'other1':otr1,
                'other2':otr2,
                'other3':otr3,
                'other4':otr4,

            }
            return render(request, 'choose_paper/analysis0.html', context)

 # 显示一篇论文的详细内容（根据所选则的论文名称）
def choose_detail(request): 
    if request.POST:
        #详情 按钮响应事件
        if 'xiangQing' in request.POST:
            node = request.POST['node']
            artical = ""
            author = ""
            abstract = ""
            affiliation = ""
            IndexTerm = ""
            keyword = ""
            inlink = ""
            outlink = ""
            GeneralTerm = ""
            year = ""
            venue = ""
            num = 0
            kk = ""
            with open('choose_paper/data/nodes2.txt', 'r', encoding='utf-8') as f:
                linef = f.readlines()
                for j in range(len(linef)):
                    line = linef[j].split()
                    node_sp = node.split()
                    flag = False
                    if(len(node_sp) == (len(line)-1)):
                        for i in range(len(node_sp)):
                            if(node_sp[i] != line[i+1]):
                                flag = False
                                break
                            else:
                                flag = True
                    if flag == True:
                        artical = line[0]
                        num = j
                        break
            with open('choose_paper/data/abstracts.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                abstract = lines[num]
            with open('choose_paper/data/affiliations.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                affiliation = lines[num]
            with open('choose_paper/data/articles.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                artical = lines[num]
            with open('choose_paper/data/authors.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                author = lines[num]
            with open('choose_paper/data/GeneralTerms.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                GeneralTerm = lines[num]
            with open('choose_paper/data/IndexTerms.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                IndexTerm = lines[num]
            with open('choose_paper/data/inlinks_nodes.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                inlink = lines[num]
            with open('choose_paper/data/keywords.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                keyword = lines[num]
            with open('choose_paper/data/outlinks_nodes.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                outlink = lines[num]
            with open('choose_paper/data/years.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                year = lines[num]
            with open('choose_paper/data/venues.txt', 'r', encoding="ISO-8859-1") as f:
                lines = f.readlines()
                venue = lines[num]
                # kk=abstract

            # 论文的引用的可视化：-----------------outlinks--------------------------------------
            # node_id 是包括该论文和其引用论文id的list
            node_name = []
            node_name = outlink.split('/')
            node_name.insert(0, str(node))
            # 构造关系图的nodes
            nodes = []
            for name in node_name:
                temp = {}
                temp['name'] = name
                temp['symbolSize'] = 20
                temp['category'] = 1
                nodes.append(temp)
            nodes[0]['category'] = 0

            # 构造关系图的links
            sname = node_name[0]
            links = []
            for i in nodes:
                links.append({"source": sname, "target": i.get("name")})

            # 构造关系图的categories
            categories = [{'name': '本论文'}, {'name': '引用论文'}]
            c = (Graph(init_opts=opts.InitOpts(theme=ThemeType.ROMA))
                 .add('', nodes=nodes, links=links, categories=categories, repulsion=1500, edge_symbol=['circle', 'arrow'])
                 .set_global_opts(title_opts=opts.TitleOpts(title="论文引用关系图")))
            c.render(path="choose_paper/data/graph/graph_outlink_tmp.txt")
            # print(c.load_javascript())
            outlink_txt = "{"
            with open('choose_paper/data/graph/graph_outlink_tmp.txt', 'r', encoding='utf-8') as f:
                line = f.readlines()
                for i in range(15, len(line)-4):
                    outlink_txt += line[i]

            # 论文的引用的可视化：-----------------inlinks--------------------------------------
            # node_id 是包括该论文和其引用论文id的list
            node_name = []
            node_name = inlink.split('/')
            node_name.insert(0, str(node))
            # 构造关系图的nodes
            nodes = []
            for name in node_name:
                temp = {}
                temp['name'] = name
                temp['symbolSize'] = 20
                temp['category'] = 1
                nodes.append(temp)
            nodes[0]['category'] = 0

            # 构造关系图的links
            sname = node_name[0]
            links = []
            for i in nodes:
                links.append({"source": sname, "target": i.get("name")})

            # 构造关系图的categories
            categories = [{'name': '本论文'}, {'name': '引用论文'}]
            c = (Graph(init_opts=opts.InitOpts(theme=ThemeType.ROMA))
                 .add('', nodes=nodes, links=links, categories=categories, repulsion=1500, edge_symbol=['circle', 'arrow'])
                 .set_global_opts(title_opts=opts.TitleOpts(title="论文引用关系图")))
            c.render(path="choose_paper/data/graph/graph_inlink_tmp.txt")
            inlink_txt = "{"
            with open('choose_paper/data/graph/graph_inlink_tmp.txt', 'r', encoding='utf-8') as f:
                line = f.readlines()
                for i in range(15, len(line)-4):
                    inlink_txt += line[i]
# --------------------------------------------------------------------------
            context = {
                'articals': artical,
                'authors ': author,
                'abstracts': abstract,
                'affiliations': affiliation,
                'IndexTerms ': IndexTerm,
                'keywords': keyword,
                'nodes': node,
                'inlinks': inlink_txt,
                'outlinks': outlink_txt,
                'GeneralTerms': GeneralTerm,
                'graph_outlink': c,
                'years': year,
                'venues': venue,

            }

            return render(request, 'choose_paper/paper_detail.html', context)
        #分析按钮 响应事件
        else:
            #获取满足的论文集合id 返回相关分析
            node = request.POST['node']
            artical = ""
            inlink = ""
            outlink = ""
            with open('choose_paper/data/nodes2.txt', 'r', encoding='utf-8') as f:
                linef = f.readlines()
                for j in range(len(linef)):
                    line = linef[j].split()
                    node_sp = node.split()
                    flag = False
                    if(len(node_sp) == (len(line)-1)):
                        for i in range(len(node_sp)):
                            if(node_sp[i] != line[i+1]):
                                flag = False
                                break
                            else:
                                flag = True
                    if flag == True:
                        artical = line[0]
                        num = j
                        break
            #论文集合(由此篇论文及其inlink，outlink组成)
            list_paper=[]
            with open('choose_paper/data/articles.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                artical = lines[num]
                list_paper.append(int(artical.strip()))
            with open('choose_paper/data/inlinks.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                inlink = lines[num].split()
                for i in range(len(inlink)):
                    list_paper.append(int(inlink[i])  )
            with open('choose_paper/data/outlinks.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                outlink = lines[num].split()
                for i in range(len(outlink)):
                    list_paper.append(int(outlink[i]))

            
            #进行训练
            dictAll0=np.load('choose_paper/PaperCompare/data_process_result/dictAll0.npy').item()
            dictAll1=np.load('choose_paper/PaperCompare/data_process_result/dictAll1.npy').item()
            dictAll2=np.load('choose_paper/PaperCompare/data_process_result/dictAll2.npy').item()
            dictAll3=np.load('choose_paper/PaperCompare/data_process_result/dictAll3.npy').item()
            dictAll4=np.load('choose_paper/PaperCompare/data_process_result/dictAll4.npy').item()
            dict0 = np.load('choose_paper/PaperCompare/data_process_result/dict0.npy').item()
            dict1 = np.load('choose_paper/PaperCompare/data_process_result/dict1.npy').item()
            dict2 = np.load('choose_paper/PaperCompare/data_process_result/dict2.npy').item()
            dict3 = np.load('choose_paper/PaperCompare/data_process_result/dict3.npy').item()  
            dict4 = np.load('choose_paper/PaperCompare/data_process_result/dict4.npy').item()
            #-------------------------------------------------model0------------------
            if('check' in request.POST):#有规则的（记得是五个！！）
                finalresult=predmodel("choose_paper/PaperCompare/model/model0.h5",list_paper,dict0,dictAll0,0)
                
            else:#无规则的
                finalresult=predmodel_norules("choose_paper/PaperCompare/model_/model0.h5",list_paper,dict0,dictAll0,0)
            result0=[]#记录最后的相似论文对的论文名字
            for tmp in finalresult:
                result_0=""
                for j in range(2):
                    with open("choose_paper/data/nodes2.txt",'r',encoding='utf-8') as f:
                      lines=f.readlines()
                      for i in range(len(lines)):
                          line=lines[i].split()
                          if(str(tmp[0][j])==str(line[0])):  
                              for m in range(1,len(line)):
                                  result_0+=line[m]
                                  result_0+=" "
                    if(j==0):
                      result_0+=","
                # result_0+=")"
                result0.append(result_0)
            other0=[]
            for tmp in finalresult:
                other0.append(tmp[1])  
            #-------------------------------------------------model1------------------
            if('check' in request.POST):#有规则的
                finalresult=predmodel("choose_paper/PaperCompare/model/model1.h5",list_paper,dict1,dictAll1,1)
            else:#无规则的
                finalresult=predmodel_norules("choose_paper/PaperCompare/model_/model1.h5",list_paper,dict1,dictAll1,1)
            result1=[]#记录最后的相似论文对的论文名字
            
            for tmp in finalresult:
                result_1=""
                for j in range(2):
                    with open("choose_paper/data/nodes2.txt",'r',encoding='utf-8') as f:
                      lines=f.readlines()
                      for i in range(len(lines)):
                          line=lines[i].split()
                          if(str(tmp[0][j])==str(line[0])):  
                              for m in range(1,len(line)):
                                  result_1+=line[m]
                                  result_1+=" "
                    if(j==0):
                      result_1+=","
                # result_1+=")"
                result1.append(result_1)
            other1=[]
            for tmp in finalresult:
                other1.append(tmp[1])
            #-------------------------------------------------model2------------------
            if('check' in request.POST):#有规则的
                finalresult=predmodel("choose_paper/PaperCompare/model/model2.h5",list_paper,dict2,dictAll2,2)
            else:#无规则的
                finalresult=predmodel_norules("choose_paper/PaperCompare/model_/model2.h5",list_paper,dict2,dictAll2,2)
            result2=[]#记录最后的相似论文对的论文名字
            
            for tmp in finalresult:
                result_2=""
                for j in range(2):
                    with open("choose_paper/data/nodes2.txt",'r',encoding='utf-8') as f:
                      lines=f.readlines()
                      for i in range(len(lines)):
                          line=lines[i].split()
                          if(str(tmp[0][j])==str(line[0])):  
                              for m in range(1,len(line)):
                                  result_2+=line[m]
                                  result_2+=" "
                    if(j==0):
                      result_2+=","
                # result_2+=")"
                result2.append(result_2)
            other2=[]
            for tmp in finalresult:
                other2.append(tmp[1])
            #-------------------------------------------------model3------------------
            if('check' in request.POST):#有规则的
                finalresult=predmodel("choose_paper/PaperCompare/model/model3.h5",list_paper,dict3,dictAll3,3)
            else:#无规则的
                finalresult=predmodel_norules("choose_paper/PaperCompare/model_/model3.h5",list_paper,dict3,dictAll3,3)
            result3=[]#记录最后的相似论文对的论文名字
            
            for tmp in finalresult:
                result_3=""
                for j in range(2):
                    with open("choose_paper/data/nodes2.txt",'r',encoding='utf-8') as f:
                      lines=f.readlines()
                      for i in range(len(lines)):
                          line=lines[i].split()
                          if(str(tmp[0][j])==str(line[0])):  
                              for m in range(1,len(line)):
                                  result_3+=line[m]
                                  result_3+=" "
                    if(j==0):
                      result_3+=","
                # result_3+=")"
                result3.append(result_3)
            other3=[]
            for tmp in finalresult:
                other3.append(tmp[1])
            #-------------------------------------------------model4------------------
            if('check' in request.POST):#有规则的
                finalresult=predmodel("choose_paper/PaperCompare/model/model4.h5",list_paper,dict4,dictAll4,4)
            else:#无规则的
                finalresult=predmodel_norules("choose_paper/PaperCompare/model_/model4.h5",list_paper,dict4,dictAll4,4)
            result4=[]#记录最后的相似论文对的论文名字
            
            for tmp in finalresult:
                result_4=""
                for j in range(2):
                    with open("choose_paper/data/nodes2.txt",'r',encoding='utf-8') as f:
                      lines=f.readlines()
                      for i in range(len(lines)):
                          line=lines[i].split()
                          if(str(tmp[0][j])==str(line[0])):  
                              for m in range(1,len(line)):
                                  result_4+=line[m]
                                  result_4+=" "
                    if(j==0):
                      result_4+=","
                # result_4+=")"
                result4.append(result_4)
            other4=[]
            for tmp in finalresult:
                # ans=[]
                # for i in range(1,len(tmp)):
                #     ans.append(tmp[i])
                other4.append(tmp[1])
            #---------------------传递----------------------
            context = {
                'back': result0,  # 获取满足年份，会议的论文id
                'issue': result1,
                'contribute': result2,
                'measure': result3,
                'exam': result4,
                'other0':other0,
                'other1':other1,
                'other2':other2,
                'other3':other3,
                'other4':other4,

            }
            return render(request, 'choose_paper/analysis0.html', context)


            # return HttpResponse(kk)


