import json
from pathlib import Path
import cv2

def verify_outputs():
    pdir = Path("dataset/processed")
    print(f"Inspecting processed directory: {pdir.resolve()}")
    all_ok = True
    count = 0
    for f in sorted(pdir.iterdir()):
        if not f.is_file() or f.name == ".gitkeep":
            continue
        count += 1
        readable = False
        details = ""
        if f.suffix == ".json":
            try:
                with open(f, "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                readable = True
                details = f"JSON keys={list(data.keys())}"
            except Exception as e:
                details = f"JSON Error: {e}"
        elif f.suffix == ".png":
            try:
                img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    readable = True
                    details = f"PNG shape={img.shape}, dtype={img.dtype}"
                else:
                    details = "cv2.imread returned None"
            except Exception as e:
                details = f"PNG Error: {e}"
        else:
            details = f"Unknown format: {f.suffix}"

        status = "OK" if readable else "FAIL"
        if not readable:
            all_ok = False
        print(f"  [{status}] {f.name:52} | Size: {f.stat().st_size:9} B | {details}")

    print("-" * 75)
    print(f"Total processed files inspected: {count} | All readable: {all_ok}")
    return all_ok

if __name__ == "__main__":
    verify_outputs()
