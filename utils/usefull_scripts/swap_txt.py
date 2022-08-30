from csv import reader
import os

a = "11"
dir_target = "dataset_webapp/exp8/labels/train"
dir_out = "dataset_webapp/txt_out"
for filename in os.listdir(dir_target):
    # print(filename)
    lines = []
    with open(os.path.join(dir_target, filename), "r") as file:
        csv_reader = reader(file, delimiter=" ")
        for row in csv_reader:
            if row[0] == "11":
                row[0] = "2"
            if int(row[0]) > int(a):
                row[0] = str(int(row[0]) - 1)

            lines.append(row)

    with open(os.path.join(dir_out, filename), mode="w") as outfile:
        for s in lines:
            res = " ".join(s)
            outfile.write("%s\n" % res)
            print(outfile)
