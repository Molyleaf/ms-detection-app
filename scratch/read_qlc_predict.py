import json
import sys

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    with open("scripts/qlc-0103.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    for i, cell in enumerate(nb.get("cells", [])):
        source = "".join(cell.get("source", []))
        if "def predict_and_evaluate(" in source:
            print(f"=== Cell {i} ===")
            lines = source.split("\n")
            for idx, line in enumerate(lines):
                if "def predict_and_evaluate(" in line:
                    for k in range(idx, min(idx + 60, len(lines))):
                        print(lines[k])
                    break
            print("=" * 60)

if __name__ == "__main__":
    main()
