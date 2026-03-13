from torchvision import transforms

# GTSRB dataset-level mean and std (precomputed from 01_eda.ipynb)
MEAN = [0.3586026519084019, 0.31745679806375865, 0.3353860326349692]
STD  = [0.2738793602624722, 0.25774299920137855, 0.26788354220321686]

def get_train_transforms(img_size=32):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])

def get_val_transforms(img_size=32):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])