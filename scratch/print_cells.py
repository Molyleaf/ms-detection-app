import json

with open("scripts/qlc-0103.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in range(10, 15):
    if idx < len(nb["cells"]):
        cell = nb["cells"][idx]
        print(f"Cell {idx} ({cell['cell_type']}):")
        source = "".join(cell.get("source", []))
        print(source[:500])
        print("=" * 60)
