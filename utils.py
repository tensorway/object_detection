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
        # print(tensor.shape)
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

def visualize_img(img, annotation, cats=["null" for i in range(1000)], color=None, width=3, font_size=None):
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
        strings.append(cats[ann['category_id']])


    img = draw_bounding_box(img, bboxes, strings, color=color, width=width, font_size=font_size)
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
            # print(idiv, jdiv, i, j, offset_i, offset_j, w, h, final_grid_size_i, final_grid_size_j)

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
    final_grid_size_i=9,
    final_grid_size_j=9,
    scaled_img_height=288,
    scaled_img_width=288,
    cats=[str(i) for i in range(1000)],
    thres = 0.5,
    ann={},
    width_bbox=1,
    font_size=None,
    transform_f = lambda img, ann: (torch.tensor(img), ann),
    batching_func = lambda l: (l[0][0], l[0][1])
):

    tens, ann = transform_f(img, ann)
    imgs, _ = batching_func([[tens, ann]])
    model.eval()
    preds_class_ids, preds_objectness, preds_offsets, preds_normalized_wh = model(imgs)
    model.train()

    objectness = preds_objectness.squeeze(0).squeeze(0)
    class_id   = th.argmax(preds_class_ids, dim=1).squeeze(0)
    class_prob = th.max(preds_class_ids, dim=1)[0].squeeze(0)
    prob = objectness     
    
    anns = []
    annotations = []
    for i in range(final_grid_size_i):
        for j in range(final_grid_size_j):
            if prob[i, j].item() > thres:
                centy = (i+preds_offsets[0, 0, i, j]).item()*scaled_img_height/final_grid_size_i
                centx = (j+preds_offsets[0, 1, i, j]).item()*scaled_img_width /final_grid_size_j
                width = th.square(preds_normalized_wh[0, 0, i, j]).item() * scaled_img_width
                height= th.square(preds_normalized_wh[0, 1, i, j]).item() * scaled_img_height
                top_left_x = centx-width/2
                top_left_y = centy-height/2
                bbox = [top_left_x, top_left_y, width, height]
                category_id = int(class_id[i, j].item())
                ann_tmp = {'bbox':bbox, 'category_id':category_id}
                annotations.append(ann_tmp)
                
    offsets    = preds_offsets.squeeze(0)
    img = transforms.ToPILImage()(img)
    small_img = transforms.Resize((scaled_img_height, scaled_img_width))(img)
    annotated_img = visualize_img(small_img, annotations, cats=cats, width=width_bbox, font_size=font_size)
    return annotated_img, (class_id, class_prob, objectness)


def yolo_visualize_tensors(class_id, class_prob, objectness):
    plt.subplot(1, 3, 1)
    plt.imshow(objectness.cpu().detach().numpy())
    plt.subplot(1, 3, 2)
    plt.imshow(class_id.cpu().detach().numpy())
    plt.subplot(1, 3, 3)
    plt.imshow(class_prob.cpu().detach().numpy())


def tensors2annotations(yolo_output, ):
    preds_class_ids, preds_objectness, preds_offsets, preds_normalized_wh = yolo_output
    objectness = preds_objectness.squeeze(0).squeeze(0)
    class_id   = th.argmax(preds_class_ids, dim=1).squeeze(0)
    class_prob = th.max(preds_class_ids, dim=1)[0].squeeze(0)
    prob = objectness     
    
    anns = []
    annotations = []
    for i in range(final_grid_size_i):
        for j in range(final_grid_size_j):
            if prob[i, j].item() > thres:
                centy = (i+preds_offsets[0, 0, i, j]).item()*scaled_img_height/final_grid_size_i
                centx = (j+preds_offsets[0, 1, i, j]).item()*scaled_img_width /final_grid_size_j
                width = th.square(preds_normalized_wh[0, 0, i, j]).item() * scaled_img_width
                height= th.square(preds_normalized_wh[0, 1, i, j]).item() * scaled_img_height
                top_left_x = centx-width/2
                top_left_y = centy-height/2
                bbox = [top_left_x, top_left_y, width, height]
                category_id = int(class_id[i, j].item())
                ann_tmp = {'bbox':bbox, 'category_id':category_id}
                annotations.append(ann_tmp)

    return annotations






