import json

with open("scripts/qlc-0103.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("scratch/qlc_code_extracted.py", "w", encoding="utf-8") as f:
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            f.write(f"\n# ==================== CELL {idx} ====================\n")
            f.write("".join(cell.get("source", [])))
            f.write("\n")

print("Extracted all code cells to scratch/qlc_code_extracted.py")
