import json
import csv

# Open the jsonl file for reading and the csv file for writing
with open('data_json/SubtaskA/subtaskA_dev_monolingual.jsonl', 'r') as json_file, open('data_csv/subtaskA_dev_monolingual.csv', 'w', newline='', encoding='utf-8') as csv_file:
    # Create a csv writer object
    writer = csv.writer(csv_file)

    # Write the header row to the csv file
    writer.writerow(['text', 'label', 'model', 'source', 'id'])

    # Loop through each line in the jsonl file
    for line in json_file:
        # Load the json data from the line
        data = json.loads(line)

        # Write the data to the csv file
        writer.writerow([data['text'], data['label'], data['model'], data['source'], data['id']])