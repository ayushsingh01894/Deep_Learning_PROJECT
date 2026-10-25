"""
convert_to_food101_layout.py
Maujooda dataset (train/<class>/*.jpg, test/<class>/*.jpg) ko project ke
preprocess.py wale layout mein badalta hai (files MOVE hoti hain, copy nahi, to extra disk nahi lagti):

    data/raw/food101/images/<class>/train_<n>.jpg , test_<n>.jpg
    data/raw/food101/meta/classes.txt, train.txt, test.txt

Run (project root se):
    python convert_to_food101_layout.py
    python convert_to_food101_layout.py "path\\to\\food101"     # agar folder alag ho
"""
import os
import shutil
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "raw", "food101")
train_dir, test_dir = os.path.join(ROOT, "train"), os.path.join(ROOT, "test")
images_dir, meta_dir = os.path.join(ROOT, "images"), os.path.join(ROOT, "meta")

for d in (train_dir, test_dir):
    if not os.path.isdir(d):
        sys.exit(f"Folder nahi mila: {d}\nPehle check karo ki {ROOT} mein train/ aur test/ hain.")
if os.path.exists(images_dir):
    sys.exit(f"'{images_dir}' pehle se hai. Script dobara chalane se pehle usse hata do ya check karo.")

classes = sorted(set(os.listdir(train_dir)) & set(os.listdir(test_dir)))
only = set(os.listdir(train_dir)) ^ set(os.listdir(test_dir))
if only:
    print("Warning: ye folders sirf ek side hain, skip honge:", sorted(only))

os.makedirs(meta_dir, exist_ok=True)
lines = {"train": [], "test": []}
for split, src in (("train", train_dir), ("test", test_dir)):
    for c in classes:
        os.makedirs(os.path.join(images_dir, c), exist_ok=True)
        files = sorted(f for f in os.listdir(os.path.join(src, c))
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        for n, f in enumerate(files):
            new = f"{split}_{n}"
            shutil.move(os.path.join(src, c, f), os.path.join(images_dir, c, new + ".jpg"))
            lines[split].append(f"{c}/{new}")
        print(f"{split:5s} {c:15s} {len(files)} images")

for name, content in (("classes.txt", classes), ("train.txt", lines["train"]), ("test.txt", lines["test"])):
    with open(os.path.join(meta_dir, name), "w") as f:
        f.write("\n".join(content) + "\n")

# khali purane folders hata do
for d in (train_dir, test_dir):
    shutil.rmtree(d, ignore_errors=True)

print(f"\nDone. Train: {len(lines['train'])}, Test: {len(lines['test'])}, Classes: {len(classes)}")
print("\npreprocess.py mein SELECTED_CLASSES is list se replace karo:\n")
print("SELECTED_CLASSES = [")
for c in classes:
    print(f'    "{c}",')
print("]")
