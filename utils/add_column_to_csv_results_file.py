import csv

csv_path = "/home/phisch/multimodal/test_results/model_scores_zero-shot.csv"
tmp_path = csv_path + ".tmp"

# Read the original CSV and add the "mode" column after "dataset"
with open(csv_path, "r", newline="") as infile, open(tmp_path, "w", newline="") as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    header = next(reader)
    # Insert "mode" after "dataset"
    if "mode" not in header:
        idx = header.index("method") + 1
        new_header = header[:idx] + ["mode"] + header[idx:]
        writer.writerow(new_header)
        for row in reader:
            # Insert empty value for mode
            new_row = row[:idx] + [""] + row[idx:]
            writer.writerow(new_row)
    else:
        # Already present, just copy
        writer.writerow(header)
        for row in reader:
            writer.writerow(row)

# Replace original file
import os
os.replace(tmp_path, csv_path)

print(f'Added "mode" column to {csv_path}')