
import torch
import os
import transforms 
from dataset import BezierDataset
from util import dict_collate_fn
from model import BenizerNet
from loss import HungarianBezierLoss
import time
from tqdm import tqdm

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
                                                    num_workers=2)
    

    # Load model
    model = BenizerNet().to(device)

    # Setup hyper parameters
    num_epochs = 400
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

    total_loss, curve_loss, classification_loss, segmentation_loss = [], [], [], []
    
    print("Start training...")

    for epoch in range(num_epochs):
        time_now = time.time()
        for input, label in tqdm(train_loader, total=train_size, desc=f"Training epoch {epoch}"):
            input, label = input.to(device), label.to(device)
            pred = model(input)
            loss, log_data = criterion(pred, label)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Record loss
            total_loss.append(loss.item())
            curve_loss.append(log_data['curve_loss'])
            classification_loss.append(log_data['classification_loss'])
            segmentation_loss.append(log_data['segmentation_loss'])

        # Save the model after each 10 epochs
        if epoch % 10 == 0:
            save_model(model, epoch)

        print(f"Epoch {epoch} finished in {(time.time() - time_now)} with loss {loss.item()} curve loss {log_data['curve_loss']} classification loss {log_data['classification_loss']} segmentation loss {log_data['segmentation_loss']}")

if __name__ == '__main__':
    main()