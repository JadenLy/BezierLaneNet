import torch

from dataset import BezierDataset
import transforms
from util import dict_collate_fn

def evaluate():

    # Load the model
    model = torch.load('model/model_400.pt')

    # Load the test set
    test_transform = transforms.Compose([
        transforms.Resize((360, 640), (360, 640), ignore_x=None), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], normalize_target=True, ignore_x=None)
    ])
    test_dataset = BezierDataset('data/test_set', 'test', transforms=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=2,
                                                collate_fn=dict_collate_fn,
                                                shuffle=False,
                                                num_workers=2)
    # Run the output
    for image in 
    pred = model()

    # Calculate metrics




if __name__ == '__main__':
    evaluate()
