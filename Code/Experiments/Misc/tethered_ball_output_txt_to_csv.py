import csv
filename = r'Fly8.1_Cr_M02_10.1_data.txt' ## write fullname of the file
f = open(filename, 'r')
new_csv_file = filename[:-4] + '.csv'
csv_file = open(new_csv_file, 'w')
writer_csv = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer_csv.writerow(['x', 'y', 'z', 'theta'])
read='no'
for lines in f:
    if 'Filtered data' in lines:
        read='yes'
        continue
    if read=='yes':
        a = lines.split(',')
        a=[float(s) for s in a]
        writer_csv.writerow(a)
f.close()
csv_file.close()