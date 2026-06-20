with open("scripts/AD_extracted.py", "r", encoding="utf-8") as f:
    code = f.read()

# Let's search for * or / or 100 in the code of ApplicabilityDomainChecker
# and print out the lines
lines = code.split("\n")
for idx, line in enumerate(lines):
    if "ApplicabilityDomainChecker" in line or idx > 400:
        if any(k in line for k in ["*", "/", "100", "%"]):
            print(f"Line {idx+1}: {line.strip()}")
