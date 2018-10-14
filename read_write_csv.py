import os
import csv

#csv 写入
stu1 = ['marry',26]
stu2 = ['bob',23]
#打开文件，追加a
out = open('Stu_csv.csv','a', newline='')
#设定写入模式
csv_write = csv.writer(out,dialect='excel')
#写入具体内容
csv_write.writerow(stu1)
csv_write.writerow(stu2)
print ("write over")

scv_file = csv.reader(open('Stu_csv.csv','r'))
print(scv_file)
for stu in scv_file:
    print(stu)
    name = stu[0]
    age = int(stu[1])
    print('name=%d,')