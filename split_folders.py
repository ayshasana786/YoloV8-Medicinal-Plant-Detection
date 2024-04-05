import splitfolders

# Split with a ratio.
# To only split into training and validation set, set a tuple to ratio, i.e, (.8, .2).
splitfolders.ratio(r"C:\Users\aysha\OneDrive\Desktop\Yolo_v8\Segmented Medicinal Leaf Images", output=r"C:\Users\aysha\OneDrive\Desktop\Yolo_v8\splitted_leaves",
    seed=1337, ratio=(.8,.2), group_prefix=None, move=False) # default values