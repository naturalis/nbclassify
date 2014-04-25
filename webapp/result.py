import sys

ip = sys.argv[1]


sections = ["Parvisepalum", "Pardalopetalum", "Paphiopedilum", "Coryopedilum", "Cochlopetalum", "Brachypetalum", "Barbata"]

infile = open("%s_out.txt"%(ip), 'r')

index = -1

file1 = infile.readlines()
lijstje = []

counter = -1

for x in file1:
	counter += 1
	try:
		if counter >= 1 and counter <= 7:
			lijstje.append(float(x.strip().strip(",").strip("'")))
	except:
		continue		
		
for y in lijstje:
	if y > 0:
		index = lijstje.index(y)
	else:
		continue

result = sections[index]


output = open("%s_result.txt"%(ip) , 'w')

output.write("%s"%(result))
output.close()