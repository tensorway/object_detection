import torch
import random
import torchvision.transforms as transforms

class MNISTDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, 
    base_dataset, 
    dim=(1, 50, 200), 
    max_digits=10, 
    minratio=0.3, 
    maxratio=3, 
    mindx=30, 
    mindy=30,
    max_iou=0.1,
    ratio = None
    ):
        super().__init__()
        self.dataset = base_dataset
        self.dim = dim
        self.options = list(range(1, max_digits+1))
        self.max_iou = max_iou
        self.mindx = mindx
        self.mindy = mindy
        self.minratio = minratio if ratio is None else ratio#ratio= y/x
        self.maxratio = maxratio if ratio is None else 1/ratio
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ret = torch.zeros(self.dim)
        bboxs = []
        labels = []
        how_many = random.choice(self.options)
        for ndigit in range(how_many):
            max_iou, max_times = 0, 2*len(self.options)
            height, width = 0, 0
            # bbox = []
            while (max_iou > self.max_iou or width<= 0)and max_times>0 :
                x1 = random.randint(0, self.dim[2]-self.mindx)
                y1 = random.randint(0, self.dim[1]-self.mindy)
                ratio = random.random()*(self.maxratio-self.minratio)+self.minratio
                if (self.dim[1]-y1)/(self.dim[2]-x1) > ratio:
                    x2 = random.randint(x1+self.mindx, self.dim[2])
                    width = x2 - x1
                    height = int(width*ratio)
                    y2 = y1 + height
                else:
                    y2 = random.randint(y1+self.mindy, self.dim[1])
                    height = y2 - y1
                    width = int(height/ratio)
                    x2 = x1 + width

                bbox = [x1, y1, width, height]
                max_iou = 0
                for bbox_cmp in bboxs:
                    iou = self.IoU(bbox, bbox_cmp)
                    if iou > max_iou:
                        max_iou = iou
                max_times -= 1
            img, labl = self.dataset[random.randint(0, len(self.dataset)-1)]
            img = transforms.Resize((height, width))(img)
            tensor = transforms.ToTensor()(img)
            ret[:, y1:y2, x1:x2] += tensor
            ret = ret*(ret>0.3)
            xr, yr, w, h = self.find_bbox(tensor)
            bboxs.append((x1+xr, y1+yr, w, h))

            labels.append(labl)
        annotation = [{'bbox':b, 'category_id':item} for b, item in zip(bboxs, labels)]
        return ret, annotation

    def find_bbox(self, img, thres=0.3):
        img = img[0]
        x1, y1 = float('inf'),float('inf')
        x2, y2 = 0, 0
        for i, row in enumerate(img):
            for j, pixel in enumerate(row):
                if pixel > thres:
                    if j < x1: x1 = j
                    if i < y1: y1 = i
                    if j > x2: x2 = j
                    if i > y2: y2 = i
        return (x1, y1, x2-x1, y2-y1)

    def IoU(self, bbox1, bbox2):
        x11, y11 = bbox1[0], bbox1[1]
        x12, y12 = x11+bbox1[2], y11+bbox1[3]
        x21, y21 = bbox2[0], bbox2[1]
        x22, y22 = x21+bbox2[2], y21+bbox2[3]
        xtop, ytop = max(x11, x21), max(y11, y21)
        xbot, ybot = min(x12, x22), min(y12, y22)

        if xtop<xbot and ytop<ybot:
            intersection = (xbot-xtop)*(ybot-ytop)
        else:
            intersection = 0
        area1 = bbox1[2]*bbox1[3]
        area2 = bbox2[2]*bbox2[3]
        union = area1 + area2 - intersection
        return intersection/union