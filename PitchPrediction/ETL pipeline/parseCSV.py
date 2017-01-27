import csv

def readCSV():
#    data = pd.read_csv('raw_data/Clayton_Kershaw.csv', names=None)
	csv_file = 'raw_data/Clayton_Kershaw.csv'
	txt_file = 'clean_data/Clayton_Kershaw'

	file = open(csv_file, "rb")
	output_file = open(txt_file, "w")
	reader = csv.reader(file)
	
	row_num = 0 
	for row in reader:
		if row_num is 0:
			header = row
		else:
			col_num = 0
			features = []
			for col in row:
				if col_num is 0:
					features.append(col)
#					print >> output_file, '%d' %(int(col)),
				else:
					features.append(str(col_num)+':'+col)
#					print >> output_file,'%d:%d' %(int(col_num), int(col)),
				col_num += 1
#			print >> output_file, '\n',
			print >> output_file, ' '.join(features)
		row_num +=1

	file.close()

def main():
    readCSV()

if __name__ == "__main__":
    main()
