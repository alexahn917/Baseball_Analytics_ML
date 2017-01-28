import csv

def readCSV():
	csv_file = 'raw_data/Clayton Kershaw_R.csv'
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
				else:
					features.append(str(col_num)+':'+col)
				col_num += 1
			print >> output_file, ' '.join(features)
		row_num +=1

	file.close()

def main():
    readCSV()

if __name__ == "__main__":
    main()
