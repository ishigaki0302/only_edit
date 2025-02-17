import csv
import json

def convert_to_csv(data, output_file):
    with open(output_file, 'w', newline='', encoding='utf-16') as csvfile:
        fieldnames = ['index', 'subject', 'relation', 'new_object', 'old_object']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, item in enumerate(data, start=1):
            writer.writerow({
                'index': i,
                'subject': item['subject'],
                'relation': item['prompt'],
                'new_object': item['target_new']['str'],
                'old_object': item['target_true']
            })

def main(data_path):
    with open(f"{data_path}.json", 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    output_file = f'{data_path}_for_excal.csv'
    convert_to_csv(json_data, output_file)

main(data_path = f"data/en2jp_data")