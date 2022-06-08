import torch

from dataset import BezierDataset
import transforms
from util import dict_collate_fn
from lane import LaneEval
from model import BenizerNet, lane_pruning
from collections import OrderedDict
from tqdm import tqdm
import os
import warnings
import cv2
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore', category=UserWarning)

def evaluate():

    # Load the model
    model = BenizerNet().cuda()
    checkpoint = torch.load('model/best_model.pt')
    checkpoint = OrderedDict((k.replace('aux_head', 'lane_classifier') if 'aux_head' in k else k, v)
                                      for k, v in checkpoint.items())
    model.load_state_dict(checkpoint, strict=True)

    # Load the test set
    test_transform = transforms.Compose([
        transforms.Resize((360, 640), (360, 640), ignore_x=None), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], normalize_target=True, ignore_x=None)
    ])
    test_dataset = BezierDataset('data/test_set', 'test', transforms=test_transform)
    test_size = len(test_dataset)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=1,
                                                collate_fn=dict_collate_fn,
                                                shuffle=False,
                                                num_workers=4)
    
    bench_eval = LaneEval()

    # Run the output
    accuracy, fp, fn = [], [], []
    for index, (image, labels) in tqdm(enumerate(test_loader), total=test_size, desc='Running evaluation'):
        image = image.to('cuda')
        labels = labels[0]
        pred = model.infer(image)[0]
        pred_widths = [[c[0] for c in lane] for lane in pred]

        a, p, n = bench_eval.bench(pred_widths, labels['lanes'], labels['h_samples'], 0)

        accuracy.append(a)
        fp.append(p)
        fn.append(n)

        # Visualize
        image = cv2.imread(os.path.join('data/test_set', test_dataset.image_files[index]))
        for lane in pred:
            lane = np.array([pts for pts in lane if pts[0] != -2]).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [lane], isClosed=False, color=(0, 0, 255), thickness=5)

        cv2.imwrite(f"vis/{index}.jpg", image)

    # Summarize

    df = pd.DataFrame({"accuracy": accuracy, "fp": fp, "fn": fn})
    df.to_csv('result.csv')
    print(f'TuSimple eval: accuracy {np.mean(accuracy) * 100} fp {np.mean(fp):.2} fn {np.mean(fn):.2f}')



if __name__ == '__main__':
    evaluate()
