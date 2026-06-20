import json
import sys

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    with open("scripts/qlc-0103.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    for idx, cell in enumerate(nb.get("cells", [])):
        source = "".join(cell.get("source", []))
        if "def remove_isotope_peaks" in source or "def normalize_intensity" in source:
            print(f"Cell {idx}:")
            lines = source.split("\n")
            for line in lines[:40]:
                print(line)
            print("=" * 60)

if __name__ == "__main__":
    main()
