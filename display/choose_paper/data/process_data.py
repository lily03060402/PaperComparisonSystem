# start=0
# with open('./articles.txt', 'r', encoding='utf-8') as a:
#     for id in a.readlines():
#         print(id.strip())
#         with open('./nodes.txt', 'r', encoding='utf-8') as f:
#             with open('./nodes2.txt', 'a', encoding='utf-8') as ff:
#                 lines=f.readlines()
#                 # print(start)
#                 for i in range(start,len(lines)):
#                     line = lines[i].strip()
#                     line = line.split()
#                     if(id.strip() == line[0]):
#                         for m in range(1,len(line)):
#                             ff.write(line[m])
#                             ff.write(" ")
#                         start+=1
#                     if(id.strip() < line[0]):
#                         break
#                 ff.write("\n")

# with open('./articles2.txt', 'a',encoding='utf-8') as ff:
#       ff.write(line.strip()) # 把末尾的'\n'删掉
#       ff.write("\n")

# with open('./inlinks.txt', 'r', encoding='utf-8') as f1:
#     for id in f1.readlines():
#         # print(id)
#         id = id.split()
#         # print(len(id))
#         with open('./inlinks_nodes.txt', 'a', encoding='utf-8') as f2:
#             if(len(id) == 0):
#                 f2.write('\n')
#                 continue
#             else:
#                 for i in range(len(id)):
#                     flag=True
#                     with open('./nodes2.txt', 'r', encoding='utf-8') as f3:
#                         for line in f3.readlines():
#                             if flag==False:
#                                 break
#                             line = line.split()
#                             if(id[i] == line[0]):
#                                 flag=False
#                                 for j in range(1, len(line)):
#                                     f2.write(line[j].strip())
#                                     f2.write(' ')
#                     f2.write("    /    ")
#                 f2.write('\n')

#用来生成id year venue 文件
with open('./articles.txt', 'r', encoding='utf-8') as f1:
        with open('./years.txt', 'r', encoding='utf-8') as f2:
            with open('./venues.txt', 'r', encoding="ISO-8859-1") as f3:
                ids=f1.readlines()
                years=f2.readlines()
                venues=f3.readlines()
                with open('./id_year_venue.txt', 'a', encoding='utf-8') as f4:
                    for i in range(len(ids)):
                        f4.write(ids[i])
                        f4.write(years[i])
                        f4.write(venues[i])
                        f4.write("\n")

     