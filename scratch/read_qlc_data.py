import json

with open("scripts/qlc-0103.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in range(13):
    cell = nb["cells"][idx]
    source = "".join(cell.get("source", []))
    if any(k in source for k in ["read_excel", "read_csv", "xlsx", "csv", "data"]):
        print(f"Cell {idx}:")
        print(source[:500])
        print("=" * 50)
