import json
import sys

def print_cell_content(path, cell_idx):
    sys.stdout.reconfigure(encoding='utf-8')
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cell = nb.get('cells', [])[cell_idx]
    source = "".join(cell.get('source', []))
    lines = source.split('\n')
    for line in lines[200:400]:
        print(line)

if __name__ == "__main__":
    print_cell_content("scripts/qlc-0103.ipynb", 16)
