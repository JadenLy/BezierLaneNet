
import torch
import os
import transforms 
from dataset import BezierDataset
from util import dict_collate_fn
from model import BenizerNet
from loss import HungarianBezierLoss
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def save_model(model, epoch):
    model_dir = 'model'
    torch.save(model, os.path.join(model_dir, f'model_{epoch}.pt'))

def main():

    # Prepare the dataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_transform = transforms.Compose([
        transforms.RandomAffine((-10, 10), translate=(50, 20), scale=(0.8, 1.2), ignore_x=None), 
        transforms.RandomHorizontalFlip(0.5, ignore_x=None), 
        transforms.Resize((360, 640), (360, 640), ignore_x=None), 
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.15), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], normalize_target=True, ignore_x=None)
    ])

    train_dataset = BezierDataset('data/train_set', transforms=train_transform)
    train_size = len(train_dataset)
    print(f"Finished loading training set with {train_size} samples")
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=10,
                                                    collate_fn=dict_collate_fn,
                                                    sampler=torch.utils.data.RandomSampler(train_dataset),
                                                    num_workers=1)
    
    # Load valid
    val_transform = transforms.Compose([
        transforms.Resize((360, 640), (360, 640), ignore_x=None), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], normalize_target=True, ignore_x=None)
    ])
    val_dataset = BezierDataset('data/train_set', 'val', transforms=val_transform)
    val_size = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size=1,
                                                collate_fn=dict_collate_fn,
                                                shuffle=False,
                                                num_workers=1)

    # Load model
    model = BenizerNet().to(device)

    # Setup hyper parameters
    num_epochs = 4
    criterion = HungarianBezierLoss()

    # Change parameters  1/10 lr for deformable offsets,
    parameters=[{'params':'conv_offset', 'lr':0.0006 * 0.1}, {'params': []}]

    for name, param in model.named_parameters():
        if 'conv_offset' in name:
            parameters[0]['params'] = param
        else:
            parameters[1]['params'].append(param)

    optimizer = torch.optim.Adam(parameters, lr=0.0006)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_dataset) * num_epochs)

    train_total_loss = []
    val_total_loss, val_curve_loss, val_clas_loss, val_seg_loss = [], [], [], []
    print("Start training...")

    for epoch in range(num_epochs):
        time_now = time.time()
        model.train()
        total_loss = 0.0
        for index, (input, labels) in enumerate(train_loader):
            input = input.to(device)
            labels = [{k: v.to(device) for k, v in label.items()} for label in labels]
            pred = model(input)
            optimizer.zero_grad()
            trainloss, _ = criterion(pred, labels)
            trainloss.backward()
            optimizer.step()
            scheduler.step()

            # Record loss
            total_loss += trainloss.item()

            print(f'Iteration {index} loss {trainloss.item()}')

        total_loss = total_loss / train_size
        train_total_loss.append(total_loss)

        # Test on valid set
        model.eval()
        total_loss = 0.0
        for image, labels in val_loader:
            image, labels = image.to(device), label.to(device)
            pred = model(image)
            loss, log_data = criterion(pred, labels)
            total_loss += loss.item()

        total_loss = total_loss / val_size
        val_total_loss.append(total_loss)

        # Save the model after each 10 epochs
        if epoch % 10 == 0:
            save_model(model, epoch)

        print(f"Epoch {epoch} finished in {(time.time() - time_now)} with train loss {train_total_loss[-1]} valid loss {total_loss}")

    # Plot loss
    plt.plot(train_total_loss, label='train total')
    plt.plot(val_curve_loss, label='val curve')
    plt.plot(val_clas_loss, label='val classification')
    plt.plot(val_seg_loss, label='val segmentation')
    plt.plot(val_total_loss, label='val total')
    plt.legend()
    plt.title('Training loss plot')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    print(f"Model Training finished")

if __name__ == '__main__':
    main()