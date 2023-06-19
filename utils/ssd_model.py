"""
第2章物体检测（SSD）
"""

# 导入软件包
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Function
import torch.utils.data as data
import torch
import cv2
import numpy as np
import os.path as osp
from itertools import product as product
from math import sqrt as sqrt

# 用于从文件和文本中读取、加工、保存XML的库  
import xml.etree.ElementTree as ET

# 文件夹“utils”的data_augumentation.py中的import。 对输入图像进行预处理的类  
from utils.data_augumentation import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

# 文件夹“utils”中描述函数match的match.py中的import  
from utils.match import match


#创建学习、验证用图像数据和标注数据的文件路径列表

def make_datapath_list(rootpath):
    """
    创建用于保存指向数据路径的列表

    Parameters
    ----------
    rootpath : str
        数据文件夹的路径

    Returns
    -------
    ret : train_img_list, train_anno_list, val_img_list, val_anno_list
        用于保存数据路径的列表
    """

    # 创建图像文件与标注文件的路径模板
    imgpath_template = osp.join(rootpath, 'JPEGImages', '%s.jpg')
    annopath_template = osp.join(rootpath, 'Annotations', '%s.xml')

    #分别取得训练和验证用的文件的ID
    train_id_names = osp.join(rootpath + 'ImageSets/Main/train.txt')
    val_id_names = osp.join(rootpath + 'ImageSets/Main/val.txt')

    #创建训练数据的图像文件与标注文件的路径列表
    train_img_list = list()
    train_anno_list = list()

    for line in open(train_id_names):
        file_id = line.strip()  #删除空格和换行符
        img_path = (imgpath_template % file_id)  #图像的路径
        anno_path = (annopath_template % file_id)  #标注的路径
        train_img_list.append(img_path)  #添加到列表中
        train_anno_list.append(anno_path)  #添加到列表中

    #创建验证数据的图像文件和标注文件的路径列表
    val_img_list = list()
    val_anno_list = list()

    for line in open(val_id_names):
        file_id = line.strip()  #删除空格和换行符
        img_path = (imgpath_template % file_id)  #图像的路径
        anno_path = (annopath_template % file_id)  #标注的路径
        val_img_list.append(img_path)  #添加到列表中
        val_anno_list.append(anno_path)  #添加到列表中

    return train_img_list, train_anno_list, val_img_list, val_anno_list


#将xml格式的标注转换为列表形式的类


class Anno_xml2list(object):
    """
    使用图像的尺寸信息，对每一张图像包含的xml格式的标注数据进行正规化处理，并保存到列表中

    Attributes
    ----------
    classes :  列表
       用于保存VOC分类名的列表
    """

    def __init__(self, classes):

        self.classes = classes

    def __call__(self, xml_path, width, height):
        """
       使用图像的尺寸信息，对每一张图像包含的xml格式的标注数据进行正规化处理，并保存到列表中。

        Parameters
        ----------
        xml_path : str
           xml文件路径
        width : int
            对象图像宽度
        height : int
            对象图像高度

        Returns
        -------
        ret : [[xmin, ymin, xmax, ymax, label_ind], ... ]
           用于保存物体的标注数据的列表。列表元素数量与图像内包含的物体数量相同
        """

        #将图像内包含的所有物体的标注保存到该列表中
        ret = []

        #读取xml文件
        xml = ET.parse(xml_path).getroot()

       #将图像内包含的物体数量作为循环次数进行迭代
        for obj in xml.iter('object'):

            #将标注中注明检测难度为difficult的对象剔除
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

          #用于保存每个物体的标注信息的列表
            bndbox = []

            name = obj.find('name').text.lower().strip()  # 物体名称
            bbox = obj.find('bndbox')  # 包围盒的信息

            # 获取标注的 xmin, ymin, xmax, ymax，并正规化为0～1的值
            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            for pt in (pts):
                #VOC的原点从(1,1)开始，因此将其减1变为（0, 0）
                cur_pixel = int(bbox.find(pt).text) - 1

                #使用宽度和高度进行正规化
                if pt == 'xmin' or pt == 'xmax':  #x方向时用宽度除
                    cur_pixel /= width
                else:  #y方向时用高度除
                    cur_pixel /= height

                bndbox.append(cur_pixel)

            #取得标注的分类名的index并添加
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

            #将res加[xmin, ymin, xmax, ymax, label_ind]
            ret += [bndbox]

        return np.array(ret)  # [[xmin, ymin, xmax, ymax, label_ind], ... ]



# 对输入图像进行预处理的类


class DataTransform():
    """
     图像和标注的预处理类。训练和推测时分别采用不同的处理
     将图像尺寸调整为 300 像素 ×300 像素
     学习时进行数据增强处理


    Attributes
    ----------
    input_size : int
        需要调整的图像大小
    color_mean : (B, G, R)
        各个颜色通道的平均值
    """

    def __init__(self, input_size, color_mean):
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(),  #将int转换为float32
                ToAbsoluteCoords(),  #返回正规化后的标注数据
                PhotometricDistort(),  #随机地调整图像的色调
                Expand(color_mean),  #扩展图像的画布尺寸
                RandomSampleCrop(),  #随机地截取图像内的部分内容
                RandomMirror(), #对图像进行翻转
               ToPercentCoords(), #将标注数据进行规范化，使其值在0~1的范围内
               Resize(input_size), #将图像尺寸调整为input_size×input_size
               SubtractMeans(color_mean) #减去BGR的颜色平均值
            ]),
            'val': Compose([
              ConvertFromInts(), #将int转换为float
              Resize(input_size), #将图像尺寸调整为input_size×input_size
              SubtractMeans(color_mean) #减去BGR的颜色平均值
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
         指定预处理的模式
        """
        return self.data_transform[phase](img, boxes, labels)


class VOCDataset(data.Dataset):
    """
    创建VOC2012的Dataset的类，继承自PyTorch的Dataset类

    Attributes
    ----------
    img_list : 列表
        保存图像路径的列表
    anno_list : リスト
        保存标注数据路径的列表
    phase : 'train' or 'test'
         用于指定是进行学习还是训练
    transform : object
          预处理类的实例
    transform_anno : object
         将xml格式的标注转换为列表的实例
    """

    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase #指定train或val
        self.transform = transform #图像的变形处理
        self.transform_anno = transform_anno #将xml的标注转换为列表

    def __len__(self):
        '''返回图像的张数'''
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        获取经过预处理的图像的张量形式的数据和标注
        '''
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def pull_item(self, index):
        '''经过预处理的图像的张量格式的数据、标注数据，获取图像的高度和宽度'''

        # 1.读入图像
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path) #[ 高度 ][ 宽度 ][ 颜色BGR]
        height, width, channels = img.shape #获取图像的尺寸

        # 2.将xml格式的标注信息转换为列表
        anno_file_path = self.anno_list[index]
        anno_list = self.transform_anno(anno_file_path, width, height)

        # 3.实施预处理
        img, boxes, labels = self.transform(
            img, self.phase, anno_list[:, :4], anno_list[:, 4])

         #由于颜色通道的顺序是BGR，因此需要转换为RGB的顺序
         #然后将（高度、宽度、颜色通道）的顺序变为（颜色通道、高度、宽度）的顺序
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)

        #创建由BBox和标签组合而成的np.array，变量名gt是ground truth（答案）的简称
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width


def od_collate_fn(batch):
    """
    从Dataset中取出的标注数据的尺寸，对于每幅图像都是不同的
    如果图像内的物体数量为两个，尺寸就是(2, 5) ；如果是三个，就会变成（3, 5）
    要创建能够处理这种不同的DataLoader，就需要对collate_fn进行定制
    collate_fn是PyTorch中从列表创建小批次数据的函数
    在保存了小批次个列表变量batch的前面加入指定小批次的编号，将两者作为一个列表对象输出
    """

    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])  # sample[0]是图像img
        targets.append(torch.FloatTensor(sample[1]))  # sample[1]是标注gt

     #imgs是小批次大小的列表
     #列表的元素是torch.Size([3, 300, 300])
     #将该列表变成torch.Size([batch_num, 3, 300, 300])的张量
     imgs = torch.stack(imgs, dim=0)

     #targets是标注数据的正解gt的列表
     #列表的大小与小批次的大小一样
     #列表targets的元素为[n, 5]
     #n对于每幅图像都是不同的，表示每幅图像中包含的物体数量
     #5是[xmin, ymin, xmax, ymax, class_index]

    return imgs, targets


#创建34层神经网络的vgg模块
def make_vgg():
    layers = []
    in_channels = 3  #颜色通道数

    #在vgg模块中使用的卷积层和最大池化等的通道数
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256,
           256, 'MC', 512, 512, 512, 'M', 512, 512, 512]

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'MC':
            #ceil模式输出的尺寸，对计算结果（float）进行向上取整
            #默认情况下输出的尺寸，对计算结果（float）进行向下取整的floor模式
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return nn.ModuleList(layers)


#创建8层网络的extras模块
def make_extras():
    layers = []
    in_channels = 1024 #从vgg模块输出，作为extras模块的输入图像的通道数

    # extras模块的卷积层的通道数的配置数据
    cfg = [256, 512, 128, 256, 128, 256, 128, 256]

    layers += [nn.Conv2d(in_channels, cfg[0], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[0], cfg[1], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[1], cfg[2], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[2], cfg[3], kernel_size=(3), stride=2, padding=1)]
    layers += [nn.Conv2d(cfg[3], cfg[4], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[4], cfg[5], kernel_size=(3))]
    layers += [nn.Conv2d(cfg[5], cfg[6], kernel_size=(1))]
    layers += [nn.Conv2d(cfg[6], cfg[7], kernel_size=(3))]

    return nn.ModuleList(layers)


#loc_layers负责输出DBox的位移值
#创建用于输出对DBox的每个分类的置信度confidence的conf_layers

def make_loc_conf(num_classes=21, bbox_aspect_num=[4, 6, 6, 6, 4, 4]):

    loc_layers = []
    conf_layers = []

   #VGG的第22层，对应conv4_3（source1）的卷积层
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[0]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[0]
                              * num_classes, kernel_size=3, padding=1)]

    #VGG的最后一层，对应（source2）的卷积层
    loc_layers += [nn.Conv2d(1024, bbox_aspect_num[1]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(1024, bbox_aspect_num[1]
                              * num_classes, kernel_size=3, padding=1)]

    #extras的对应（source3）的卷积层
    loc_layers += [nn.Conv2d(512, bbox_aspect_num[2]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(512, bbox_aspect_num[2]
                              * num_classes, kernel_size=3, padding=1)]

    #extras的对应（source4）的卷积层
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[3]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[3]
                              * num_classes, kernel_size=3, padding=1)]

    # extraの（source5）に対する畳み込み層
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[4]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[4]
                              * num_classes, kernel_size=3, padding=1)]

    #extras的对应（source5）的卷积层
    loc_layers += [nn.Conv2d(256, bbox_aspect_num[5]
                             * 4, kernel_size=3, padding=1)]
    conf_layers += [nn.Conv2d(256, bbox_aspect_num[5]
                              * num_classes, kernel_size=3, padding=1)]

    return nn.ModuleList(loc_layers), nn.ModuleList(conf_layers)


#对convC4_3的输出进行scale=20的L2Norm的正规化处理的层
class L2Norm(nn.Module):
    def __init__(self, input_channels=512, scale=20):
        super(L2Norm, self).__init__() #调用父类的构造函数
        self.weight = nn.Parameter(torch.Tensor(input_channels))
        self.scale = scale #系数weight的初始值
        self.reset_parameters() #对参数进行初始化
        self.eps = 1e-10

    def reset_parameters(self):
        '''将连接参数设置为大小为scale的值，执行初始化'''
        init.constant_(self.weight, self.scale) #weight的值全部设为scale（=20）

    def forward(self, x):
       '''对38×38的特征量，求512个通道的平方和的根值
       使用38×38个值，对每个特征量进行正规化处理后再乘以系数的层'''

        #对每个通道进行38x38个特征量的通道方向的平方和计算
        #接下来进行正规化处理
        #norm的张量尺寸为torch.Size([batch_num, 1, 38, 38])
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)

        #乘以系数。每个通道1个系数，总共有512个系数
        #因为self.weight的张量尺寸是torch.Size([512])
        #转换为torch.Size([batch_num, 512, 38, 38])
        weights = self.weight.unsqueeze(
            0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = weights * x

        return out


#输出DBox的类
class DBox(object):
    def __init__(self, cfg):
        super(DBox, self).__init__()

        #初始化设置
        self.image_size = cfg['input_size'] #图像尺寸为300像素
        #[38, 19, …] 每个source的特征量图的大小
        self.feature_maps = cfg['feature_maps']
        self.num_priors = len(cfg["feature_maps"]) #source的个数=6
        self.steps = cfg['steps'] #[8, 16, …]DBox的像素尺寸
        self.min_sizes = cfg['min_sizes']#[30, 60, …]小正方形的DBox的像素尺寸
        self.max_sizes = cfg['max_sizes'] #[60, 111, …]大正方形的DBox的像素尺寸
        self.aspect_ratios = cfg['aspect_ratios'] #长方形的DBox的纵横比

    def make_dbox_list(self):
        '''创建DBox'''
        mean = []
        # 'feature_maps': [38, 19, 10, 5, 3, 1]
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):  # 创建到f为止的2对排列组合f_P_2 个
                #特征量的图像尺寸
               # 300 / 'steps': [8, 16, 32, 64, 100, 300],
                f_k = self.image_size / self.steps[k]

               #DBox的中心坐标x,y　但是，正规化为0～1的值
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                #宽高比为1的小DBox [cx,cy, width, height]
                # 'min_sizes': [30, 60, 111, 162, 213, 264]
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                #宽高比为1的大DBox [cx,cy, width, height]
                # 'max_sizes': [45, 99, 153, 207, 261, 315],
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                #其他宽高比的DBox [cx,cy, width, height]
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

       #将DBox转换成张量torch.Size([8732, 4])
        output = torch.Tensor(mean).view(-1, 4)

        #为防止DBox的大小超出图像范围，将尺寸调整为最小为0，最大为1
        output.clamp_(max=1, min=0)

        return output


#使用位移信息，将DBox转换成BBox的函数
def decode(loc, dbox_list):
    """
    使用位移信息，将DBox转换成BBox

    Parameters
    ----------
    loc:  [8732,4]
        用SSD模型推测位移信息
    dbox_list: [8732,4]
        DBox的信息

    Returns
    -------
    boxes : [xmin, ymin, xmax, ymax]
        BBox的信息
    """

    #DBox以[cx, cy, width, height]形式被保存
    #loc以[Δcx, Δcy, Δwidth, Δheight]形式被保存

    #从位移信息求取BBox
    boxes = torch.cat((
        dbox_list[:, :2] + loc[:, :2] * 0.1 * dbox_list[:, 2:],
        dbox_list[:, 2:] * torch.exp(loc[:, 2:] * 0.2)), dim=1)
    #boxes的尺寸为torch.Size([8732, 4])

    #BBox的坐标信息从[cx, cy, width, height]变为[xmin, ymin, xmax, ymax]
    boxes[:, :2] -= boxes[:, 2:] / 2 #变换为坐标(xmin,ymin)
    boxes[:, 2:] += boxes[:, :2] #变换为坐标(xmax,ymax)

    return boxes

#进行Non−Maximum Suppression处理的函数


def nm_suppression(boxes, scores, overlap=0.45, top_k=200):
    """
    进行Non−Maximum Suppression处理的函数
    将boxes中过于重叠的BBox删除

    Parameters
    ----------
    boxes : [ 超过了置信度阈值（0.01）的BBox数量,4]
          BBox信息
    scores : [ 超过了置信度阈值（0.01）的BBox数量 ]
          conf的信息

    Returns
    -------
    keep :列表
          保存按conf降序通过了nms处理的index
    count ：int
          通过了nms处理的BBox的数量
    """

    #创建return的雏形
     count = 0
     keep = scores.new(scores.size(0)).zero_().long()
     #keep ：torch.Size([超过了置信度阈值的BBox数量])，元素全部为0

    #计算各个BBox的面积area
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    #复制boxes，准备用于稍后进行BBox的过重叠度IoU计算时使用的雏形
    tmp_x1 = boxes.new()
    tmp_y1 = boxes.new()
    tmp_x2 = boxes.new()
    tmp_y2 = boxes.new()
    tmp_w = boxes.new()
    tmp_h = boxes.new()

    #将socre按升序排列
    v, idx = scores.sort(0)

    #将前面top_k个(200个)BBox的index取出（也有不到200个的情况）
    idx = idx[-top_k:]

    #当idx的元素数量不为0时，则执行循环
    while idx.numel() > 0:
        i = idx[-1]  #将现在conf最大的index赋值给i

     #将conf最大的index保存到keep中现在最末尾的位置
     #开始删除该index的BBox和重叠较大的BBox
        keep[count] = i
        count += 1

        #当处理到最后一个BBox时，跳出循环
        if idx.size(0) == 1:
            break

        #keep中保存了目前的conf最大的index,因此将idx减1
        idx = idx[:-1]

        # -------------------
        #开始对keep中保存的BBox和重叠较大的BBox抽取出来并删除
        # -------------------
        #到减去1的idx为止，将BBox放到out指定的变量中
        torch.index_select(x1, 0, idx, out=tmp_x1)
        torch.index_select(y1, 0, idx, out=tmp_y1)
        torch.index_select(x2, 0, idx, out=tmp_x2)
        torch.index_select(y2, 0, idx, out=tmp_y2)

       #对所有的BBox，当前的BBox=index被到i为止的值覆盖（clamp）
        tmp_x1 = torch.clamp(tmp_x1, min=x1[i])
        tmp_y1 = torch.clamp(tmp_y1, min=y1[i])
        tmp_x2 = torch.clamp(tmp_x2, max=x2[i])
        tmp_y2 = torch.clamp(tmp_y2, max=y2[i])

        #将w和h的张量尺寸设置为index减去1后的结果
        tmp_w.resize_as_(tmp_x2)
        tmp_h.resize_as_(tmp_y2)

        #对clamp处理后的BBox求高度和宽度
        tmp_w = tmp_x2 - tmp_x1
        tmp_h = tmp_y2 - tmp_y1

       #如果高度或宽度为负数，则设为0
        tmp_w = torch.clamp(tmp_w, min=0.0)
        tmp_h = torch.clamp(tmp_h, min=0.0)

        #计算经过clamp处理后的面积
        inter = tmp_w*tmp_h

        #IoU = intersect部分/[area(a) + area(b) – intersect部分]的计算
        rem_areas = torch.index_select(area, 0, idx) #各个BBox的原有面积
        union = (rem_areas - inter) + area[i] #对两个区域的面积求与
        IoU = inter/union

        #只保留IoU比overlap小的idx
         idx = idx[IoU.le(overlap)] #le是进行Less than or Equal to处理的逻辑运算
         #IoU比overlap大的idx，与刚开始选择并保存到keep中的idx对相同的物体进行了
         #BBox包围，因此要删除

    #while跳出循环体，结束执行

    return keep, count


#从SSD的推测时的conf和loc的输出数据，得到消除了重叠的BBox并输出


class Detect(Function):

    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45):
        self.softmax = nn.Softmax(dim=-1)#准备使用Softmax函数对conf进行正规化处理
        self.conf_thresh = conf_thresh #只处理conf高于conf_thresh=0.01的DBox
        self.top_k = top_k #对conf最高的top_k个进行nm_supression计算时使用,top_k = 200
         self.nms_thresh = nms_thresh #进行nm_supression计算时，如果IoU比nms_ thresh=0.45大，就认为是同一物                                                       #体的BBox

    def forward(self, loc_data, conf_data, dbox_list):
        """
        执行正向传播计算

        Parameters
        ----------
        loc_data:  [batch_num,8732,4]
            位移信息
        conf_data: [batch_num, 8732,num_classes]
            检测的置信度
        dbox_list: [8732,4]
           DBox的信息

        Returns
        -------
        output : torch.Size([batch_num, 21, 200, 5])
            （batch_num、分类、conf的top200、BBox的信息）
        """

       #获取各个尺寸
        num_batch = loc_data.size(0) #最小批的尺寸
        num_dbox = loc_data.size(1) #DBox的数量= 8732
        num_classes = conf_data.size(2)  #分类数量= 21

        # 使用Softmax对conf进行正规化处理
        conf_data = self.softmax(conf_data)

        #生成输出数据对象。张量尺寸为[minibatch数, 21, 200, 5]
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)

       #将cof_data从[batch_num,8732,num_classes]调整为[batch_num, num_classes,8732]
        conf_preds = conf_data.transpose(2, 1)

        #按最小批进行循环处理
        for i in range(num_batch):

            # 1.从loc和DBox求取修正过的BBox [xmin, ymin, xmax, ymax]
            decoded_boxes = decode(loc_data[i], dbox_list)

           #创建conf的副本
            conf_scores = conf_preds[i].clone()

           #图像分类的循环（作为背景分类的index=0不进行计算，从index=1开始）
            for cl in range(1, num_classes):

                #2.抽出超过了conf阈值的BBox
                #创建用来表示是否超过了conf阈值的掩码
                #将阈值超过conf的索引赋值给c_mask
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                #gt表示Greater Than。超过阈值gt返回1，未超过则返回0
                # conf_scores:torch.Size([21, 8732])
                # c_mask:torch.Size([8732])

                 #scores是torch.Size([超过阈值的BBox的数量])
                 scores = conf_scores[cl][c_mask]

              #如果不存在超过阈值的conf，即当scores=[]时,则不做任何处理
                if scores.nelement() == 0:   #用nelement求取要素的数量
                    continue

                #对c_mask进行转换，使其能适用于decoded_boxes的大小
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                # l_mask:torch.Size([8732, 4])

                #将l_mask用于decoded_boxes
                boxes = decoded_boxes[l_mask].view(-1, 4)
               #decoded_boxes[l_mask]调用会返回一维列表
               #因此用view转换为（超过阈值的BBox数， 4）的尺寸

                # 3.开始Non−Maximum Suppression处理，消除重叠的BBox
                ids, count = nm_suppression(
                    boxes, scores, self.nms_thresh, self.top_k)
               #ids ：用于保存按conf降序排列通过了Non−Maximum Suppression处理的index
               #count ：通过了Non−Maximum Suppression处理的BBox的数量

                #将通过了Non−Maximum Suppression处理的结果保存到output中
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1),
                                                   boxes[ids[:count]]), 1)

        return output  # torch.Size([1, 21, 200, 5])

#创建SSD类


class SSD(nn.Module):

    def __init__(self, phase, cfg):
        super(SSD, self).__init__()

        self.phase = phase  #创建SSD类
        self.num_classes = cfg["num_classes"]  #分类数=21

       #创建SSD神经网络
        self.vgg = make_vgg()
        self.extras = make_extras()
        self.L2Norm = L2Norm()
        self.loc, self.conf = make_loc_conf(
            cfg["num_classes"], cfg["bbox_aspect_num"])

        # DBox作成
        dbox = DBox(cfg)
        self.dbox_list = dbox.make_dbox_list()

        #推测模式下，需要使用Detect类
        if phase == 'inference':
            self.detect = Detect()

    def forward(self, x):
        sources = list()  #保存source1～source6作为loc和conf的输入数据
        loc = list()  #用于保存loc的输出数据
        conf = list()  #用于保存conf的输出数据

        #计算到vgg的conv4_3
        for k in range(23):
            x = self.vgg[k](x)

       #将conv4_3的输出作为L2Norm的输入，创建source1，并添加到sources中
        source1 = self.L2Norm(x)
        sources.append(source1)

        #计算至vgg的末尾，创建source2，并添加到sources中
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)

        sources.append(x)

        #计算extras的conv和ReLU
        #并将source3～source6添加到sources中
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:  # conv→ReLU→cov→ReLUをしたらsourceに入れる
                sources.append(x)

       #对source1～source6分别进行一次卷积处理
       #使用zip获取for循环的多个列表的元素
       #需要处理source1～source6的数据，因此循环6次
        for (x, l, c) in zip(sources, self.loc, self.conf):
            #使用Permute调整要素的顺序
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
            #用l(x)和c(x)进行卷积操作
            #l(x)和c(x)的输出尺寸是[batch_num, 4*宽高比的种类数, featuremap
            #的高度,featuremap的宽度] 
            #不同source的宽高比的种类数量不同，处理比较麻烦，因此对顺序进行调整
            #使用Permute调整要素的顺序
            #增加到[minibatch的数量, featuremap的数量, featuremap的数量,
            #4*宽高比的种类数]
            #（注释）
            #torch.contiguous()是在内存中连续的设置元素的命令
            #稍后使用view函数
            #为了能够执行view函数，对象的变量在内存中必须是被连续存储的

           #对loc和conf进行变形
           #loc的尺寸是torch.Size([batch_num, 34928])
           #conf的尺寸是torch.Size([batch_num, 183372])
            loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
             conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # 进一步对loc和conf进行对齐
        # locのサイズは、torch.Size([batch_num, 8732, 4])
        # confのサイズは、torch.Size([batch_num, 8732, 21])
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)

        # 最后输出结果
        output = (loc, conf, self.dbox_list)

        if self.phase == "inference":  #推测模式
            #执行Detect类的forward
            #返回值的尺寸是torch.Size([batch_num, 21, 200, 5])
            return self.detect(output[0], output[1], output[2])

        else:  #学习模式
            return output
           #返回值是(loc, conf, dbox_list)的元组


class MultiBoxLoss(nn.Module):
    """SSD的损失函数的类"""

    def __init__(self, jaccard_thresh=0.5, neg_pos=3, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.jaccard_thresh = jaccard_thresh #0.5，match函数的jaccard系数的阈值
        self.negpos_ratio = neg_pos #3:1，难分样本挖掘的正负比例
        self.device = device #指定使用CPU或GPU进行计算

    def forward(self, predictions, targets):
        """
        损失函数的计算

        Parameters
        ----------
        predictions :  SSD网络训练时的输出 ( 元组 )
            (loc=torch.Size([num_batch, 8732, 4]), conf=torch.Size([num_batch, 8732, 21]), dbox_list=torch.Size [8732,4])。

        targets : [num_batch, num_objs, 5]
          5表示正确答案的标注信息[xmin, ymin, xmax, ymax, label_ind]

        Returns
        -------
        loss_l : 张量
              loc的损失值
        loss_c :张量
              conf的损失值

        """

       #由于SSD模型的输出数据类型是元组，因此要将其分解
        loc_data, conf_data, dbox_list = predictions

        #把握元素的数量
         num_batch = loc_data.size(0) #小批量的尺寸
         num_dbox = loc_data.size(1) #DBox的数量 = 8732
         num_classes = conf_data.size(2) #分类数量 = 21

        #创建变量，用于保存损失计算中使用的对象
        # conf_t_label ：将最接近正确答案的BBox的标签保存到各个DBox中
        # loc_t:将最接近正确答案的BBox的位置信息保存到各个DBox中
         conf_t_label = torch.LongTensor(num_batch, num_dbox).to(self.device)
         loc_t = torch.Tensor(num_batch, num_dbox, 4).to(self.device)

           #在loc_t和conf_t_label中保存
           #经过match处理的DBox和正确答案标注targets的结果
           for idx in range(num_batch): #以小批量为单位进行循环

           #获取当前的小批量的正确答案标注的BBox和标签
           truths = targets[idx][:, :-1].to(self.device) # BBox
            #标签[物体1的标签, 物体2的标签, …]
            labels = targets[idx][:, -1].to(self.device)

           #用新的变量初始化DBox变量
            dbox = dbox_list.to(self.device)

            #执行match函数，更新loc_t和conf_t_label的内容
            #（详细）
            #loc_t:保存各个DBox中最接近正确答案的BBox的位置信息
            #conf_t_label ：保存各个DBox中最接近正确答案的BBox的标签
            #但是，如果与最接近的BBox之间的jaccard重叠小于0.5
            #将正确答案BBox的标签conf_t_label设置为背景分类0
            variance = [0.1, 0.2]
            #这个variance是从DBox转换到BBox的修正计算公式中的系数
            match(self.jaccard_thresh, truths, dbox,
            variance, labels, loc_t, conf_t_label, idx)

        # ----------
        #位置的损失 ：计算loss_l
        #使用Smooth L1函数计算损失。这里只计算那些发现了物体的DBox的位移
        # ----------
        #生成用于获取检测到物体的BBox的掩码
        pos_mask = conf_t_label > 0  # torch.Size([num_batch, 8732])

        #将pos_mask的尺寸转换为loc_data
         pos_idx = pos_mask.unsqueeze(pos_mask.dim()).expand_as(loc_data)

         #获取Positive DBox的loc_data和监督数据loc_t
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

       #对发现了物体的Positive DBox的位移信息loc_t进行损失（误差）计算
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

         # ----------
         #分类预测的损失 ：计算loss_c
         #使用交叉熵误差函数进行损失计算。但是，由于绝大多数 DBox 的正确答案为背景分
         #类，因此要进行难分样本挖掘处理，将发现物体的 DBox 和背景分类 DBox 的比例
         #调整为1:3
         #然后，从预测为背景分类的DBox中，将损失值小的那些从分类预测的损失中去除
          # ----------
          batch_conf = conf_data.view(-1, num_classes)

       #计算分类预测的损失函数(设置reduction='none'，不进行求和计算，不改变维度)
        loss_c = F.cross_entropy(
            batch_conf, conf_t_label.view(-1), reduction='none')

        # -----------------
        #现在开始创建Negative DBox中，用于计算难分样本挖掘处理抽出数据的掩码
         # -----------------

         #将发现了物体的Positive DBox的损失设置为0
         #注意 ：物体标签大于1，标签0是背景
         num_pos = pos_mask.long().sum(1, keepdim=True) #以小批量为单位，对物体分类
                                                                                        #进行预测的数量
         loss_c = loss_c.view(num_batch, -1) # torch.Size([num_batch, 8732])
          loss_c[pos_mask] = 0 #将发现了物体的 DBox 的损失设置为 0

        #开始进行难分样本挖掘处理
        #计算用于对每个DBox的损失值大小loss_c进行排序的idx_rank
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)

        # （注释）
        #这里的实现代码比较特殊，不容易直观理解
        #上面两行代码是对每个DBox的损失值的大小的顺序，
        #用变量idx_rank来表示，这样就可以进行快速的访问。
        #
        #将DBox的损失值按降序排列，并将DBox的降序的index保存到loss_idx中。
        #计算用于对损失值大小loss_c进行排序用的idx_rank。
         #在这里，
        #如果要将按降序排列的索引数组loss_idx转换为从0到8732升序排列，
         #应该使用loss_idx的第几个索引值呢？idx_rank表示的就是该索引值。
         #例如，
         #要求idx_rank的第0个元素= idx_rank[0]，loss_idx的值为0的元素，
         #即loss_idx[?}=0，?表示要求取的是第几位。在这里就是? = idx_rank[0]。
         #这里，loss_idx[?]=0中的0表示原有的loss_c的第0个元素。
         #也就是说，?表示的是，求取原有的loss_c第0位的元素，在按降序排列的
         #loss_idx中是第几位这一结果
         #? = idx_rank[0] 表示loss_c的第0位元素，如果按降序排列是第几位
         #决定背景的DBox数量num_neg。通过难分样本挖掘处理后，
         #设为发现了物体的DBox的数量num_pos的三倍（self.negpos_ratio倍）。
         #但是，如果超过了DBox的数量，就将DBox数量作为上限值。
         num_neg = torch.clamp(num_pos*self.negpos_ratio, max=num_dbox)
         #idx_rank表示每个DBox的损失值按从大到小的顺序是第几位
          #生成用于读取比背景的DBox数num_neg排位更低的DBox
          # torch.Size([num_batch, 8732])
          neg_mask = idx_rank < (num_neg).expand_as(idx_rank)

        # -----------------
        #（结束）现在开始创建从Negative DBox中，用于求取难分样本挖掘抽出的数据的掩码
        # -----------------

        #转换掩码的类型，合并到conf_data中
        #pos_idx_mask是获取Positive DBox的conf的掩码
         #neg_idx_mask是获取使用难分样本挖掘提取的Negative DBox的conf的掩码
        # pos_mask：torch.Size([num_batch, 8732])→pos_idx_mask：torch.Size([num_batch, 8732, 21])
        pos_idx_mask = pos_mask.unsqueeze(2).expand_as(conf_data)
        neg_idx_mask = neg_mask.unsqueeze(2).expand_as(conf_data)

        # 从conf_data中将pos和neg取出，保存到conf_hnm中。类型是torch.
        # Size([num_pos+num_neg, 21])
         conf_hnm = conf_data[(pos_idx_mask+neg_idx_mask).gt(0)].view(-1, num_classes)
          #（注释）gt是greater than (>)的简写。这样就能取出mask为1的index。
          #虽然pos_idx_mask+neg_idx_mask是加法运算，但是只是对给index的mask进行集中
          #也就是说，无论是pos还是neg，只要是掩码为1就进行加法运算，合并成一个列表，
          #这使用gt取得

          #同样地，从作为监督数据的conf_t_label中取出pos和neg，放到conf_t_label_hnm中
           #类型是torch.Size([pos+neg])
          conf_t_label_hnm = conf_t_label[(pos_mask+neg_mask).gt(0)]

           #confidence的损失函数的计算（求元素的总和=sum）
            loss_c = F.cross_entropy(conf_hnm, conf_t_label_hnm, reduction='sum')

           #使用发现了物体的BBox的数量N（整个小批量的合计）对损失进行除法运算
           N = num_pos.sum()
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c

