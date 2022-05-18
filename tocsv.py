import csv            
import pandas as pandasForSortingCSV
  
path = 'Z:/Research_Group/Personal/Kim/piglet/r3det-on-mmdetection-master/work_dirs/r3det_r101_fpn_2x_20220308/submission/Task1_piglet.txt'
write_path = 'Z:/Research_Group/Personal/Kim/SORT/csv/submission/horizontal_boxes.csv'

f = open(write_path, "w+")
f.close()

with open(path) as f:
    for line in f.readlines():
        s = line.split(' ')
        s[9] = s[9][:-1]

        if float(s[1]) >= 0.6:
            x1 = float(s[2])
            x2 = float(s[4])
            x3 = float(s[6])
            x4 = float(s[8])

            y1 = float(s[3])
            y2 = float(s[5])
            y3 = float(s[7])
            y4 = float(s[9])
            
            xmin = min([x1, x2, x3, x4])
            xmax = max([x1, x2, x3, x4])
            ymin = min([y1, y2, y3, y4])
            ymax = max([y1, y2, y3, y4])
            
            w = [s[0], -1,xmin, ymin, xmax, ymax, s[1]]

            with open(write_path, 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(w)
                

# assign dataset
csvSort = pandasForSortingCSV.read_csv(write_path, header = None)
# sort data frame
csvSort.sort_values(csvSort.columns[0], axis=0, inplace=True)

csvSort.to_csv(write_path, index = False, header = False)
