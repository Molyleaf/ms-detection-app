import json

def main():
    with open("scripts/qlc-0103.ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    for i, cell in enumerate(nb.get("cells", [])):
        source = "".join(cell.get("source", []))
        if "70" in source or "95" in source or "confidence" in source.lower():
            print(f"Cell {i} ({cell.get('cell_type')}):")
            for line in source.split("\n"):
                if any(x in line for x in ["70", "95", "confidence", "置信"]):
                    print("  ", line.strip())
            print("-" * 50)

if __name__ == "__main__":
    main()
