import datasets

data=datasets.load_dataset("ShapeNet/ShapeNetCore",split="train")

for row in data:
    print(row)
    break