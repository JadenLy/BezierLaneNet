import numpy as np
from scipy.special import comb
import torch
from torch.utils.data._utils.collate import default_collate_err_msg_format, np_str_obj_array_pattern
import collections.abc
from scipy.optimize import linear_sum_assignment

# Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
# https://github.com/pytorch/vision/pull/3383
def upcast(t):    
    if t.is_floating_point():
        return t if t.dtype in (torch.float32, torch.float64) else t.float()
    else:
        return t if t.dtype in (torch.int32, torch.int64) else t.int()

class BezierCurve(object):
    def __init__(self, degree) -> None:
        self.degree = degree
    
    """
    The Bernstein polynomial of n, i as a function of t
    """
    def bernstein_poly(self, i, n, t):
        return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

    """
    Given a set of control points, return the
    bezier curve defined by the control points.

    points should be a list of lists, or list of tuples
    such as [ [1,1], 
                [2,3], 
                [4,5], ..[Xn, Yn] ]
    nTimes is the number of time steps, defaults to 1000

    See http://processingjs.nihongoresources.com/bezierinfo/
    """
    def bezier_fit(self, x, y):
        nPoints = len(x)
        t = np.linspace(0.0, 1.0, self.degree)
        
        polynomial_array = np.array([self.bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])
        
        xvals = np.dot(x, polynomial_array)
        yvals = np.dot(y, polynomial_array)

        return xvals, yvals

    

class BezierSampler(object):
    def __init__(self, num_points=100, degree=3):
        self.num_points = num_points
        self.degree = degree
        self.bernstein_matrix = self.get_bernstein_matrix()

    """
    The Bernstein polynomial of n, i as a function of t
    """
    def bernstein_poly(self, n, t, k):
        return t ** k * (1 - t) ** (n - k) * comb(n, k)

    def get_bezier_coefficient(self, t):
        return [[self.bernstein_poly(self.degree, ti, k) for k in range(self.degree + 1)] for ti
                                  in t]

    def get_bernstein_matrix(self):
        t = torch.linspace(0, 1, self.num_points)
        c_matrix = torch.tensor(self.get_bezier_coefficient(t))
        return c_matrix

    # Sample the points from the curve
    def sample_points(self, keypoints):
        if keypoints.numel() == 0:
            return keypoints

        if self.bernstein_matrix.device != keypoints.device:
            self.bernstein_matrix = self.bernstein_matrix.to(keypoints.device)

        return upcast(self.bernstein_matrix).matmul(upcast(keypoints))


    def bezier_to_coordinates(self, control_points, existence, resize_shape, ppl=56, gap=10):
        # control_points: L x N x 2
        H, W = resize_shape
        coordinates = []
        for i in range(len(existence)):
            if not existence[i]:
                continue

            # Find x for TuSimple's fixed y eval positions (suboptimal)
            bezier_threshold = 5.0 / H
            h_samples = np.array([1.0 - (ppl - i) * gap / H for i in range(ppl)], dtype=np.float32)
            sampled_points = self.bernstein_matrix.cpu().numpy().dot(control_points[i]) 
            temp = []
            dis = np.abs(np.expand_dims(h_samples, -1) - sampled_points[:, 1])
            idx = np.argmin(dis, axis=-1)
            for i in range(ppl):
                h = H - (ppl - i) * gap
                if dis[i][idx[i]] > bezier_threshold or sampled_points[idx[i]][0] > 1 or sampled_points[idx[i]][0] < 0:
                    temp.append([-2, h])
                else:
                    temp.append([sampled_points[idx[i]][0] * W, h])
            coordinates.append(temp)

        return coordinates

# Validate points
@torch.no_grad()
def get_valid_points(points):
    # ... x 2
    if points.numel() == 0:
        return torch.tensor([1], dtype=torch.bool, device=points.device)
    return (points[..., 0] > 0) * (points[..., 0] < 1) * (points[..., 1] > 0) * (points[..., 1] < 1)


string_classes = (str, bytes)
int_classes = int
container_abcs = collections.abc

# To keep each image's label as separate dictionaries, default pytorch behaviour will stack each key
# Only modified one line of the pytorch 1.6.0 default collate function
def dict_collate_fn(batch):
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return dict_collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return batch  # !Only modified this line
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(dict_collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [dict_collate_fn(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))

@torch.no_grad()
def cubic_bezier_curve_segment(control_points, sample_points):
    # Cut a batch of cubic bezier curves to its in-image segments (assume at least 2 valid sample points per curve).
    # Based on De Casteljau's algorithm, formula for cubic bezier curve is derived by:
    # https://stackoverflow.com/a/11704152/15449902
    # control_points: B x 4 x 2
    # sample_points: B x N x 2
    if control_points.numel() == 0 or sample_points.numel() == 0:
        return control_points
    B, N = sample_points.shape[:-1]
    valid_points = get_valid_points(sample_points)  # B x N, bool
    t = torch.linspace(0.0, 1.0, steps=N, dtype=sample_points.dtype, device=sample_points.device)

    # First & Last valid index (B)
    # Get unique values for deterministic behaviour on cuda:
    # https://pytorch.org/docs/1.6.0/generated/torch.max.html?highlight=max#torch.max
    t0 = t[(valid_points + torch.arange(N, device=valid_points.device).flip([0]) * valid_points).max(dim=-1).indices]
    t1 = t[(valid_points + torch.arange(N, device=valid_points.device) * valid_points).max(dim=-1).indices]

    # Generate transform matrix (old control points -> new control points = linear transform)
    u0 = 1 - t0  # B
    u1 = 1 - t1  # B
    transform_matrix_c = [torch.stack([u0 ** (3 - i) * u1 ** i for i in range(4)], dim=-1),
                          torch.stack([3 * t0 * u0 ** 2,
                                       2 * t0 * u0 * u1 + u0 ** 2 * t1,
                                       t0 * u1 ** 2 + 2 * u0 * u1 * t1,
                                       3 * t1 * u1 ** 2], dim=-1),
                          torch.stack([3 * t0 ** 2 * u0,
                                       t0 ** 2 * u1 + 2 * t0 * t1 * u0,
                                       2 * t0 * t1 * u1 + t1 ** 2 * u0,
                                       3 * t1 ** 2 * u1], dim=-1),
                          torch.stack([t0 ** (3 - i) * t1 ** i for i in range(4)], dim=-1)]
    transform_matrix = torch.stack(transform_matrix_c, dim=-2).transpose(-2, -1).unsqueeze(1).expand(B, 2, 4, 4)

    # Matrix multiplication
    res = transform_matrix.matmul(control_points.permute(0, 2, 1).unsqueeze(-1))  # B x 2 x 4 x 1

    return res.squeeze(-1).permute(0, 2, 1)


class HungarianMatcher(object):
    def __init__(self, degree, sampled_points, alpha, k):
        self.degree = degree
        self.sampled_points = sampled_points
        self.alpha = alpha
        self.k = k
        self.bezier_sampler = BezierSampler()
    
    # Match the $sampled_points$ points from pred to target
    @torch.no_grad()
    def match(self, logits, curves, targets):
        B, Q = logits.shape
        target_keypoints = torch.cat([i['keypoints'] for i in targets], dim=0)  # G x N x 2
        target_sample_points = torch.cat([i['sample_points'] for i in targets], dim=0)  # G x num_sample_points x 2

        # Valid bezier segments
        target_keypoints = cubic_bezier_curve_segment(target_keypoints, target_sample_points)
        target_sample_points = self.bezier_sampler.sample_points(target_keypoints)

        G, _ = target_keypoints.shape[:2]
        sizes = [target['keypoints'].shape[0] for target in targets]

        # 1. Local maxima prior
        _, max_indices = torch.nn.functional.max_pool1d(logits.unsqueeze(1),
                                                        kernel_size=self.k, stride=1,
                                                        padding=(self.k - 1) // 2, return_indices=True)
        max_indices = max_indices.squeeze(1)  # B x Q
        indices = torch.arange(0, Q, dtype=logits.dtype, device=logits.device).unsqueeze(0).expand_as(max_indices)
        local_maxima = (max_indices == indices).flatten().unsqueeze(-1).expand(-1, G)  # BQ x G

        # Safe reshape
        out_lane = curves.flatten(end_dim=1)  # BQ x N x 2

        # 2. Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - prob[target class].
        # Then 1 can be omitted due to it is only a constant.
        # For binary classification, it is just prob (understand this prob as objectiveness in OD)
        cost_label = logits.flatten().unsqueeze(-1).expand(-1, G)  # BQ x G

        # 3. Compute the curve sampling cost
        cost_curve = 1 - torch.cdist(self.bezier_sampler.sample_points(out_lane).flatten(start_dim=-2),
                                     target_sample_points.flatten(start_dim=-2),
                                     p=1) / self.sampled_points  # BQ x G

        # Bound the cost to [0, 1]
        cost_curve = cost_curve.clamp(min=0, max=1)

        # Final cost matrix (scipy uses min instead of max)
        C = local_maxima * cost_label ** (1 - self.alpha) * cost_curve ** self.alpha
        C = -C.view(B, Q, -1).cpu()

        # Hungarian (weighted) on each image
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]

        # Return (pred_indices, target_indices) for each image
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    @staticmethod
    def get_src_permutation_idx(indices):
        # Permute predictions following indices
        # 2-dim indices: (dim0 indices, dim1 indices)
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        image_idx = torch.cat([src for (src, _) in indices])

        return batch_idx, image_idx