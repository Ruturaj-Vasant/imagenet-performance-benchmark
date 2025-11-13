import os, shutil, csv, sys
root = sys.argv[1] if len(sys.argv) > 1 else "tiny-imagenet-200"
train, val = os.path.join(root,"train"), os.path.join(root,"val")
# Flatten train
for cls in os.listdir(train):
    imgdir = os.path.join(train, cls, "images")
    if os.path.isdir(imgdir):
        for f in os.listdir(imgdir):
            if f.lower().endswith((".jpeg",".jpg",".png")):
                shutil.move(os.path.join(imgdir,f), os.path.join(train,cls,f))
        shutil.rmtree(imgdir, ignore_errors=True)
# Build val/<cls> from annotations
ann = os.path.join(val, "val_annotations.txt")
with open(ann) as fh:
    for fn, cls, *_ in csv.reader(fh, delimiter="\t"):
        os.makedirs(os.path.join(val, cls), exist_ok=True)
        src = os.path.join(val, "images", fn)
        dst = os.path.join(val, cls, fn)
        if os.path.exists(src):
            shutil.move(src, dst)
shutil.rmtree(os.path.join(val,"images"), ignore_errors=True)
print("Prepared:", root)
