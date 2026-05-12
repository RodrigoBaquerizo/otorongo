import re

with open("scripts/refresh_data.py", "r", encoding="utf-8") as f:
    code = f.read()

# Make the changes in the target block (from line 630 to end)
def replace_agnostic(text):
    text = text.replace("df_atp", "df_target")
    text = text.replace("new_atp", "new_matches")
    text = text.replace("atp_from", "target_from")
    text = text.replace("atp_known_torneos", "target_known_torneos")
    text = text.replace("df_hist_atp", "df_hist_target")
    text = text.replace("missing_in_atp26", "missing_in_target")
    text = text.replace("atp_added", "target_added")
    # Fix pts_map bug
    text = text.replace("pts_map", "rankings_idx")
    
    # Update print statements
    text = text.replace("result['atp_added']", "result.get('target_added', 0)")
    text = text.replace("ATP 2026:  +{result['target_added']}", "Target:  +{result.get('target_added', 0)}")
    return text

lines = code.split("\n")
# The logic starts around line 620
start_idx = 0
for i, line in enumerate(lines):
    if "2. ACTUALIZACIÓN ARCHIVO ESPECÍFICO (ATP o Challenger)" in line:
        start_idx = i
        break

if start_idx > 0:
    block = "\n".join(lines[start_idx:])
    new_block = replace_agnostic(block)
    # Reassemble
    lines[start_idx:] = new_block.split("\n")

with open("scripts/refresh_data.py", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("Python refactor complete.")
