
import torch
import os
import transforms 
from dataset import BezierDataset
from util import dict_collate_fn
from model import BenizerNet
from loss import HungarianBezierLoss
import time
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning)

def save_model(model, filename):
    model_dir = 'focal'
    torch.save(model.state_dict(), os.path.join(model_dir, filename))

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
                                                    batch_size=50,
                                                    collate_fn=dict_collate_fn,
                                                    sampler=torch.utils.data.RandomSampler(train_dataset),
                                                    num_workers=8)
    
    # Load valid
    val_transform = transforms.Compose([
        transforms.Resize((360, 640), (360, 640), ignore_x=None), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], normalize_target=True, ignore_x=None)
    ])
    val_dataset = BezierDataset('data/train_set', 'val', transforms=val_transform)
    val_size = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                batch_size=50,
                                                collate_fn=dict_collate_fn,
                                                shuffle=False,
                                                num_workers=8)

    # Load model
    model = BenizerNet().to(device)

    # Setup hyper parameters
    num_epochs = 200
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

    val_total_loss, val_curve_loss, train_total_loss, train_curve_loss = [], [], [], []
    print("Start training...")

    best_loss = torch.inf
    for epoch in range(num_epochs):
        time_now = time.time()
        model.train()
        total_loss = 0.0
        total_curve_loss = 0.0
        for input, labels in train_loader:
            input = input.to(device)
            labels = [{k: v.to(device) for k, v in label.items()} for label in labels]
            pred = model(input)
            optimizer.zero_grad()
            trainloss, train_log = criterion(pred, labels)
            trainloss.backward()
            optimizer.step()
            scheduler.step()

            # Record loss
            total_loss += trainloss.item()
            total_curve_loss += train_log['curve_loss'].cpu().item()

            # print(f"Iteration loss {trainloss.item()}")

        total_loss = total_loss / train_size
        train_total_loss.append(total_loss)
        train_curve_loss.append(total_curve_loss / train_size)

        # Test on valid set
        model.eval()
        total_loss = 0.0
        total_curve_loss = 0.0
        for image, labels in val_loader:
            image = image.to(device)
            labels = [{k: v.to(device) for k, v in label.items()} for label in labels]
            pred = model(image)
            loss, log_data = criterion(pred, labels)
            total_loss += loss.item()
            total_curve_loss += log_data['curve_loss'].cpu().item()

        total_loss = total_loss / val_size
        val_total_loss.append(total_loss)
        val_curve_loss.append(total_curve_loss / val_size)

        # Save the model 
        if total_loss < best_loss:
            save_model(model, 'best_model.pt')

        if epoch % 10 == 0:
            save_model(model, f'model_{epoch}.pt')

        print(f"Epoch {epoch} finished in {(time.time() - time_now):.2f} seconds with train loss {train_total_loss[-1]} valid loss {total_loss}")
    
    # Plot loss
    plt.plot(train_total_loss, label='train total')
    plt.plot(val_total_loss, label='val total')
    plt.plot(train_curve_loss, label='train curve')
    plt.plot(val_curve_loss, label='val curve')
    plt.legend()
    plt.title('Training loss plot')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.savefig('plot.png')
    print(f"Model Training finished")

if __name__ == '__main__':
    main()