---
layout: post
title: Generating Video with Segmentation Masks
date:   2022-10-01 13:30:52 -0600
categories: PyTorch segmentation video python
---

This will be a bit of a long read. But most likely worth it! This is a tutorial describing how to generate videos with masks overlaid from a trained CNN. 

This tutorial assumes you have a trained model with a .pth file holding the trained weights. I trained my model using multiple GPUs so there is a segment of code 
in the following strip that has an option to put it into a `torch.nn.DataParallel()` object

There's a couple commandline options using the click interface. <br>
`results_dir`: where to save the results and where the config file detailing the 
model name is. <br>
`checkpoint_pth`: where the pth file is holding the trained model. <br>
`dataset`: the name of the dataset. <br>
`save_dir`: where the saved data corresponding to `dataset` is in .npy form 
is.<br> 

This tutorial also assumes that the images you have are stills from a video that are <em>in order numerically</em>. 

So you should have a file structure of images like: 
{% highlight html %}
images/
├─ 1.png
├─ 2.png
├─ 3.png
├─ ...
gts/
├─ 1.png
├─ 2.png
├─ 3.png
├─ ...
{% endhighlight %}

Through the code an output dir containing the image will be generated as:
{% highlight html %}
outputs/
├─ 1.png
├─ 2.png
├─ 3.png
├─ ...
{% endhighlight %}

And here is the code!

{% highlight python %}
import os 
import cv2
import torch
import click
import yaml
import time
import numpy as np 
import progressbar 

from model.FusionNetwork import NewFusionNetwork
from data.dataset import get_tDatasetImageTest
import matplotlib.pyplot as plt 


from pathlib import Path
from imageio import imwrite

def justGetBoundarySingleImage(
    img_path: str,
    output_path: str = None,
):
    """
    Takes a binary mask as input, @img_path, (just black + white) and returns 
    the boundary as a numpy array. Also saves it to the @output_path. 

    Taken from: https://medium.com/@rootaccess/how-to-detect-edges-of-a-mask-in-python-with-opencv-4bcdb3049682
    """
    img_data = cv2.imread(img_path)
    img_data = img_data > 128

    img_data = np.asarray(img_data[:, :, 0], dtype=np.double)
    gx, gy = np.gradient(img_data)
    temp_edge = gy * gy + gx * gx
    temp_edge[temp_edge != 0.0] = 255.0
    temp_edge = np.asarray(temp_edge, dtype=np.uint8)
    
    if output_path != None:
        cv2.imwrite(output_path, temp_edge)
    return temp_edge

def overlayOriginalImage(
    img_path: str,
    msk_path: str,
    maskColor: str = 'white', # white, red, purple
    output_path: str = None,
):
    msk_edge = justGetBoundarySingleImage(msk_path)

    og_image = cv2.imread(img_path)
    msk_image = cv2.imread(msk_path, cv2.IMREAD_COLOR)

    # uncomment this line 
    if maskColor == 'white':
        pass
    elif maskColor == 'red':
        msk_image[np.where((msk_image==[255, 255, 255]).all(axis=2))] = [0, 0, 255]
    elif maskColor == 'green':
        msk_image[np.where((msk_image==[255, 255, 255]).all(axis=2))] = [255, 255, 0]
    elif maskColor == 'lighter_pink':
        msk_image[np.where((msk_image==[255, 255, 255]).all(axis=2))] = [255, 0, 0]
    elif maskColor == 'pink':
        msk_image[np.where((msk_image==[255, 255, 255]).all(axis=2))] = [255, 0, 255]
    elif maskColor == 'grey':
        msk_image[np.where((msk_image==[255, 255, 255]).all(axis=2))] = [255,165,0]
    else:
        raise ValueError(f'maskColor: {maskColor} not available.')


    # this puts on the annotation or prediction with some transparency
    merged_image = cv2.addWeighted(og_image,1.0,msk_image,0.2,1.0)
    
    
    # now put on the border with no transparency / full opacity 
    merged_image = cv2.addWeighted(merged_image,1.0, cv2.cvtColor(msk_edge, cv2.COLOR_GRAY2RGB),1.0,1.0)

    if output_path != None:
        cv2.imwrite(output_path, merged_image)

def getSortedListOfFilePaths(dir: str):
    """
    returns list of files in @dir from os.listdir + the whole path to that file
    """
    list = os.listdir(dir)
    list = [x[:-4] for x in list]
    list = sorted([int(x) for x in list])
    list = [dir + '/' + str(x) + '.png' for x in list]
    return list

def getSortedListofFileNames(dir: str):
    """
    returns list of files from os.listdir just sorted (inc order) assuming ALL 
    numbers 
    same as above but just doesnt return the total path to the file it just 
    returns the file name 
    """
    list = os.listdir(dir)
    list = [x[:-4] for x in list]
    list = sorted([int(x) for x in list])
    list = [str(x) + '.png' for x in list]
    return list

def generate3Plot(
    img_path: str = "1_img.png",
    imgann_path: str = "1_imgann.png",
    imgprd_path: str = "1_img_prd.png",
    save_name: str = '1.png',
):
    """
    taking the images, image+pred masks overlays, image+ann masks overlays, 
    generates a 3 plot of each and saves them each to a directory 
    """

    img = cv2.imread(img_path, cv2.IMREAD_COLOR); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ann = cv2.imread(imgann_path, cv2.IMREAD_COLOR); ann = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
    prd = cv2.imread(imgprd_path, cv2.IMREAD_COLOR); prd = cv2.cvtColor(prd, cv2.COLOR_BGR2RGB)

    plt.style.use('seaborn-white')

    # note super titles (Original, Annotaiton, Predicted) DO NOT WORK (they cut off)
    # will have to add these manually if we want them though I think its easy
    # to work around this 
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
    for i, ax in enumerate(axs.flatten()):
        plt.sca(ax)
        if i == 0:
            plt.imshow(img, cmap=plt.cm.jet)
            # plt.title('Original')
        elif i == 1:
            plt.imshow(ann, cmap=plt.cm.jet)
            # plt.title('Annotation')
        elif i == 2:
            plt.imshow(prd, cmap=plt.cm.jet)
            # plt.title('Predicted')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_name)

@click.command(help='')
@click.option('--results_dir', type=str, default='results/DataParallel/DataParallel_11')
@click.option('--checkpoint_pth', type=str, default='results/DataParallel/DataParallel_11/current_checkpoints/DataParallel-218.pth')
@click.option('--dataset', type=str, default='CVC_ClinicDB')
@click.option('--save_dir', type = str, default='seg/data/totals/CVC_ClinicDB/')
def main(
    model_name,
    dataset,
    results_dir,
    checkpoint_pth,
    save_dir,
):
    final_cfg = yaml.load(open(Path(results_dir) / "final_config.yml", "r"),
        Loader=yaml.FullLoader)
    
    model_name = final_cfg['model_name']
    cnn_model_cfg = final_cfg['cnn_model_cfg']
    trans_model_cfg = final_cfg['trans_model_cfg']
    
    if model_name == 'NewFusionNetwork':
        model = NewFusionNetwork(
            cnn_model_cfg,
            trans_model_cfg,
            cnn_pretrained=False,
            with_fusion=True,
        ).cuda()
    else:
        raise ValueError(f'invalid model: {model_name}')

    num_gpu = torch.cuda.device_count()
    print(f'Number of GPUs: {num_gpu}')
    for i in range(num_gpu):
        print(f'Device name: {torch.cuda.get_device_name(i)}')
    if num_gpu > 1:
        model = torch.nn.DataParallel(model)

    print(f'Resuming at path: {os.path.basename(checkpoint_pth)}')
    checkpoint = torch.load(checkpoint_pth)
    model.load_state_dict(checkpoint['model_state_dict'])

    # model.load_state_dict(torch.load(checkpoint_pth))
    model.cuda()
    model.eval()

    save_path = results_dir + '/tests/' + dataset + '/'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + '/images/', exist_ok=True)
    os.makedirs(save_path + '/gts/', exist_ok=True)
    os.makedirs(save_path + '/outputs/', exist_ok=True)
    print(f'os.save_path: {save_path}')

    test_loader = get_tDatasetImageTest(
        image_root = save_dir + "/data_dataset_list_ordered.npy",
        gt_root = save_dir + "/mask_dataset_list_ordered.npy",
        normalize_gt = False,
        batch_size = 1,
        normalization = "vit",
        num_workers = 4, 
        pin_memory=True,
        originalTxtFile= save_dir + "/dataset_list_ordered.txt", # if it's not in the save_dir you have to move it there
    )
    with progressbar.ProgressBar(max_value=len(test_loader)) as bar:
        for i, image_gts in enumerate(test_loader):
            time.sleep(0.1)
            bar.update(i)
            
            images, gts, og_text = image_gts
            print(f'og_text: {og_text[0]}')
            images = images.cuda()
            gts = gts.cuda()

            with torch.no_grad():
                output = model(images)

            # output = model(images)
            output = output.sigmoid().data.cpu().numpy().squeeze()
            
            name = dataset + '_' + str(i)
            output_name = name + '_output.jpg'
            image_name = name + '_image.jpg'
            gt_name = name + '_gt.jpg'

            images = images.cpu().numpy().squeeze()
            gts = gts.cpu().numpy().squeeze().squeeze()
            images = np.transpose(images, (1,2,0))
            imwrite(save_path + '/outputs/' + og_text[0], output)
            imwrite(save_path + '/images/' + og_text[0], images)
            imwrite(save_path + '/gts/' + og_text[0], gts)

    os.makedirs(save_dir + '/img_ann/', exist_ok=True)
    os.makedirs(save_dir + '/img_prd/', exist_ok=True)
    os.makedirs(save_dir + '/MultiPlot/', exist_ok=True)

    img_names = getSortedListofFileNames(save_path + '/images/') # all file names are the same {1, 2, ... x.png}
    img_paths = getSortedListOfFilePaths(save_path + '/images/')
    ann_paths = getSortedListOfFilePaths(save_path + '/gts/')
    prd_paths = getSortedListOfFilePaths(save_path + '/outputs/')

    with progressbar.ProgressBar(max_value=len(img_names)) as bar:
        for i in range(len(img_paths)):
            time.sleep(0.1)
            bar.update(i)

            # do it for the img + anns
            overlayOriginalImage(
                img_path=img_paths[i],
                msk_path=ann_paths[i],
                maskColor='green',
                output_path=save_dir + '/img_ann/' + img_names[i],
            )
            # now for the img + prds
            overlayOriginalImage(
                img_path=img_paths[i],
                msk_path=prd_paths[i],
                maskColor='pink',
                output_path=save_dir + '/img_prd/' + img_names[i],
            )
            # now that we have img_anns + img_preds images saved, now MultiPlot
            generate3Plot(
                img_path=img_paths[i],
                imgann_path=save_dir + '/img_ann/' + img_names[i],
                imgprd_path=save_dir + '/img_prd/' + img_names[i],
                save_name=save_dir + '/MultiPlot/' + img_names[i]
            )

    


if __name__ == '__main__':
    main()
{% endhighlight %}

You can then take the images from img_prd (which is the prediction overlayed the
original image) or the images from img_ann (which is the annotation overlayed the
original image) or the images from MultiPlot which are the original image along
with the img_prd and img_ann and plug them into a gif creator. 

For the CVC-ClinicDB polyp dataset, putting the images from MultiPlot would look like this:  
![](/imgs/CVC_ColonDB_MultiPlot.gif "Input")

I used imgflip.com.

Thanks for reading!