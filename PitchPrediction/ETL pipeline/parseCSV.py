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
			for col in row:
				if col_num is 0:
					print >> output_file, '%d' %(int(col)),
				else:
					print >> output_file,'%d:%d' %(int(col_num), int(col)),
				col_num += 1
			print >> output_file, '\n',
		row_num +=1

	file.close()

"""
	text_list = []

	with open(csv_file, "r") as my_input_file:
	    for line in my_input_file:
	        line = line.split(",")
	        text_list.append(" ".join(line))

	with open(txt_file, "w") as my_output_file:
	    for line in text_list:
	        my_output_file.write("  " + line)
	    print('File Successfully written.')
"""
def main():
    readCSV()

if __name__ == "__main__":
    main()
