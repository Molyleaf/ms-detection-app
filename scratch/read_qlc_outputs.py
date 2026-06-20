import json
import sys

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    with open("scripts/qlc-0103.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "markdown":
            print(f"Cell {i}:")
            print("".join(cell.get("source", [])))
            print("=" * 60)

if __name__ == "__main__":
    main()
