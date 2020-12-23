import torch
import torchvision.transforms as transforms
import torch as th
import random
import matplotlib.pyplot as plt


def draw_bounding_box(tensor, bboxes, strings, color=None, width=1, font_size=None):
    from PIL import Image, ImageDraw, ImageFont
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(
        1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(im)
    img_h = tensor.shape[1]
    if font_size is None:
        font_size = int(25/600*img_h)
    font = ImageFont.truetype("font.otf", font_size)

    if color is None:
        all_colors = [(i*255, j*255, k*255) for i in range(2) for j in range(2) for k in range(2)]
        colors = random.choices(all_colors, k=len(bboxes))
    elif len(color) == 3 and isinstance(color[0], int):
        colors = [color]*len(bboxes)
    else:
        colors = color

    for bbox, color, string in zip(bboxes, colors, strings):
        textbox_width = int(font_size*len(string)*20/25)
        textbox_height = font_size+3
        y2 = bbox[1]-textbox_height
        if y2 < 0:
            y2 = bbox[1]+textbox_height
        fillbbox = [bbox[0], bbox[1], bbox[0]+textbox_width, y2]
        textbox = (bbox[0], min(bbox[1], y2))
        textcolor = tuple(255-c for c in color)  

        draw.rectangle(bbox, outline=color, width=width)
        draw.rectangle(fillbbox, fill=color, width=width)
        draw.text(textbox, string, fill=textcolor, font=font)

    from numpy import array as to_numpy_array
    return torch.from_numpy(to_numpy_array(im))



def visualize_img(img, annotation, cats=[str(i) for i in range(1000)], color=None, width=3, font_size=None, visualize=True):
    if type(img) != type(torch.tensor([0])):
        img = transforms.ToTensor()(img)
    if img.shape[0] < 3:
        img = torch.cat((img, img, img))
    bboxes = []
    strings = []
    for ann in annotation:
        ann2 = ann['bbox']
        ann2 = ann2[0], ann2[1], ann2[0]+ann2[2], ann2[1]+ann2[3]
        bboxes.append(ann2)
        s = cats[ann['category_id']]
        try:
            s += " --" + str(int(ann['confidence']*1000)/10)+"%"
        except:
            pass
        strings.append(s)

    img = draw_bounding_box(img, bboxes, strings, color=color, width=width, font_size=font_size)
    if visualize:
        plt.imshow(img.numpy())
    return img.numpy()



def save(model, name, major, minor):
    PATH = 'pretrained_models/' +name+'_'+str(major)+'_'+str(minor)+'.th'
    torch.save(model.state_dict(), PATH)
def load(model, name, major, minor):
    PATH = 'pretrained_models/' +name+'_'+str(major)+'_'+str(minor)+'.th'
    model.load_state_dict(torch.load(PATH))



def bce(targets, preds, lmbda, indicator=None, numsc = 1e-8):
    t1 =  -(lmbda*targets*th.log(preds+numsc) + (1-targets)*th.log(1-preds+numsc))
    if indicator is None:
        return t1.mean()
    return (t1*indicator).sum()/indicator.sum()



def batching_func(l, n_classes, n_anchors, final_grid_size_i, final_grid_size_j, last_kernel_dim, device=torch.device('cpu')):
    import math
    imgs = [x[0].unsqueeze(0) for x in l]
    imgs = torch.cat(imgs, dim=0)
    w, h = imgs.shape[-1], imgs.shape[-2]
    idiv = h/final_grid_size_i
    jdiv = w/final_grid_size_j
    class_ids     = torch.zeros(len(imgs), n_classes,   final_grid_size_i, final_grid_size_j)
    objectness    = torch.zeros(len(imgs), n_anchors,   final_grid_size_i, final_grid_size_j)
    offsets       = torch.zeros(len(imgs), n_anchors*2, final_grid_size_i, final_grid_size_j)
    normalized_wh = torch.zeros(len(imgs), n_anchors*2, final_grid_size_i, final_grid_size_j)
    lasti = 0

    annotations = [x[1] for x in l]
    for el, annotation in enumerate(annotations):
        for ann in annotation:
            bbox = ann['bbox']
            class_id = ann['category_id']
            celli = (bbox[1]+bbox[3]/2)/idiv
            cellj = (bbox[0]+bbox[2]/2)/jdiv
            i = int(celli)
            j = int(cellj)
            offset_i = celli - i
            offset_j = cellj - j
            normalized_w = math.sqrt(bbox[2]/w)
            normalized_h = math.sqrt(bbox[3]/h)

            class_ids[el, class_id, i, j] = 1
            objectness[el, lasti, i, j] = 1
            offsets[el, lasti*2:(lasti+1)*2, i, j] = torch.tensor([offset_i, offset_j], dtype=torch.float)
            normalized_wh[el, lasti*2:(lasti+1)*2, i, j] = torch.tensor([normalized_w, normalized_h], dtype=torch.float)
            lasti = (lasti+1)%n_anchors
    imgs = imgs.to(device)
    class_ids, objectness, offsets, normalized_wh = class_ids.to(device), objectness.to(device), offsets.to(device), normalized_wh.to(device)
    return imgs, (class_ids, objectness, offsets, normalized_wh)



def yolo_visualize(
    model,
    img,
    scaled_img_height=None,
    scaled_img_width=None,
    cats=[str(i) for i in range(1000)],
    thres_detection = 0.85,
    thres_non_max = 0.6,
    annotations=None,
    width_bbox=1,
    color=None,
    font_size=None,
    visualize=True,
    device=torch.device('cuda')
    ):
    if type(img) != type(th.tensor([1])) and img is not None:
        img = transforms.ToTensor()(img) #.to(device))
    img = img.to(device)
    if scaled_img_height is None or scaled_img_width is None:
        scaled_img_height = img.shape[-2]
        scaled_img_width = img.shape[-1]
    if annotations is None:
        imgs = img.unsqueeze(0)
        model.eval()
        yolo_output = model(imgs)
        model.train()
        annotations = tensors2annotations(yolo_output, thres_detection, scaled_img_height, scaled_img_width)
        annotations = non_max_supression(annotations, thres=thres_non_max)[0]
                
    img = transforms.ToPILImage()(img.cpu())
    small_img = transforms.Resize((scaled_img_height, scaled_img_width))(img)
    annotated_img = visualize_img(small_img, annotations, cats=cats, width=width_bbox, font_size=font_size, color=color, visualize=visualize)
    return annotated_img


def yolo_visualize_tensors(class_id, class_prob, objectness):
    plt.subplot(1, 3, 1)
    plt.imshow(objectness.cpu().detach().numpy())
    plt.subplot(1, 3, 2)
    plt.imshow(class_id.cpu().detach().numpy())
    plt.subplot(1, 3, 3)
    plt.imshow(class_prob.cpu().detach().numpy())


def tensors2annotations(
    yolo_output, 
    thres = 0.85,
    scaled_img_height = 288,
    scaled_img_width = 288
    ):
    bpreds_class_ids, bpreds_objectness, bpreds_offsets, bpreds_normalized_wh = yolo_output
    batch_annotations = []
    for tmp in zip(bpreds_class_ids, bpreds_objectness, bpreds_offsets, bpreds_normalized_wh):
        preds_class_ids, preds_objectness, preds_offsets, preds_normalized_wh = tmp

        objectness = preds_objectness.squeeze(0)
        class_id   = th.argmax(preds_class_ids, dim=0)
        class_prob = th.max(preds_class_ids, dim=0)[0]
        final_grid_size_i = int(objectness.shape[0])
        final_grid_size_j = int(objectness.shape[1])
        
        annotations = []
        for i in range(final_grid_size_i):
            for j in range(final_grid_size_j):
                if objectness[i, j].item() > thres:
                    centy = (i+preds_offsets[0, i, j]).item()*scaled_img_height/final_grid_size_i
                    centx = (j+preds_offsets[1, i, j]).item()*scaled_img_width /final_grid_size_j
                    width = th.square(preds_normalized_wh[0, i, j]).item() * scaled_img_width
                    height= th.square(preds_normalized_wh[1, i, j]).item() * scaled_img_height
                    top_left_x = centx-width/2
                    top_left_y = centy-height/2
                    bbox = [top_left_x, top_left_y, width, height]
                    category_id = int(class_id[i, j].item())
                    confidence = class_prob[i, j].item()*objectness[i, j].item()
                    ann_tmp = {'bbox':bbox, 'category_id':category_id, 'confidence':confidence}
                    annotations.append(ann_tmp)
        batch_annotations.append(annotations)
    return batch_annotations



def IoU(bbox1, bbox2):
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
    union = area1 + area2 - intersection + 1e-9
    return intersection/union



def non_max_supression(annotations, thres=0.6):
    torets = []
    for annotation in annotations:
        for i1, ann1 in enumerate(annotation):
            for i2, ann2 in enumerate(annotation):
                if i1 != i2:
                    iou = IoU(ann1['bbox'], ann2['bbox'])
                    if iou > thres:
                        rem = ann1 if ann1['confidence']<ann2['confidence'] else ann2
                        rem['confidence'] = -1
                        rem['bbox'] = (0, 0, 1, 1)
        toret = []
        for ann in annotation:
            if ann['confidence'] != -1:
                toret.append(ann)
        torets.append(toret)
    return torets


