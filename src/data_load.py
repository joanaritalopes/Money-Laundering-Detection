# Load and join csv files (HI and LI) by transactions, accounts and patterns

import csv
from typing import List

def load_csv(input_paths: List[str], output_path: str):
    '''Loads csv transactions.'''
    header = False
    with open(output_path, mode='w', newline='') as output_file:
        writer = None
        for input_path in input_paths:
            with open(input_path, mode='r', newline='') as file:
                reader = csv.DictReader(file)
                if not header:
                    writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    header = True
                for row in reader:
                    writer.writerow(row)


paths = {
    'data/processed/accounts.csv': ['data/raw/HI-Small_accounts.csv', 'data/raw/LI-Small_accounts.csv'],
    'data/processed/transactions.csv': ['data/raw/HI-Small_Trans.csv', 'data/raw/LI-Small_Trans.csv']
}

for out, inp in paths.items():
    load_csv(inp, out)
