"""
第2章在SSD上将预测结果画成图像的类别

"""
import numpy as np
import matplotlib.pyplot as plt 
import cv2  # OpenCV库  
import torch

from utils.ssd_model import DataTransform


class SSDPredictShow():
    """在SSD上的预测和图像的显示集中进行的类"""

    def __init__(self, eval_categories, net):
        self.eval_categories = eval_categories  #类名
        self.net = net  #SSD网络

        color_mean = (104, 117, 123)  #(BGR)颜色的平均值
        input_size = 300  # 将图像的input大小设为300×300
        self.transform = DataTransform(input_size, color_mean)  # 预处理类

    def show(self, image_file_path, data_confidence_level):
        """
        表示物体检测的预测结果的函数。。

        Parameters
        ----------
        image_file_path:  str
            图像的文件路径
        data_confidence_level: float
            预测发现的确信度阈值

        Returns
        -------
        没有。 显示在rgb_img中加入了物体检测结果的图像。  
        """
        rgb_img, predict_bbox, pre_dict_label_index, scores = self.ssd_predict(
            image_file_path, data_confidence_level)

        self.vis_bbox(rgb_img, bbox=predict_bbox, label_index=pre_dict_label_index,
                      scores=scores, label_names=self.eval_categories)

    def ssd_predict(self, image_file_path, data_confidence_level=0.5):
        """
        SSD预测函数。

        Parameters
        ----------
        image_file_path:  strt
            图像的文件路径

        dataconfidence_level: float
            预测发现的确信度阈值

        Returns
        -------
        rgb_img, true_bbox, true_label_index, predict_bbox, pre_dict_label_index, scores
        """

        # 获取rgb图像数据
        img = cv2.imread(image_file_path)  # [高度][宽度][颜色BGR]
        height, width, channels = img.shape  # 获取图像的尺寸
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 图像的预处理
        phase = "val"
        img_transformed, boxes, labels = self.transform(
            img, phase, "", "")  # 因为不存在注解，所以写成“”。
        img = torch.from_numpy(
            img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)

        # SSD预测  
        self.net.eval()  # 将网络转换为推理模式
        x = img.unsqueeze(0)  # 小批量：torch.Size([1, 3, 300, 300])

        detections = self.net(x)
       # detections的形式是torch.Size([1,21,200,5])※200是top_k的值  

        # confidence level提取标准以上
        predict_bbox = []
        pre_dict_label_index = []
        scores = []
        detections = detections.cpu().detach().numpy()

        #提取条件以上的值
        find_index = np.where(detections[:, 0:, :, 0] >= data_confidence_level)
        detections = detections[find_index]
        for i in range(len(find_index[1])):  # 把提取的物体循环几分钟
            if (find_index[1][i]) > 0:  # 非背景类的东西
                sc = detections[i][0]  # 确信度
                bbox = detections[i][1:] * [width, height, width, height]
                # find_index是迷你批次数量、类别、top的tuple  
                lable_ind = find_index[1][i]-1
                #(注释)  
                #背景类为0，减1  

                # 添加到返回值列表
                predict_bbox.append(bbox)
                pre_dict_label_index.append(lable_ind)
                scores.append(sc)

        return rgb_img, predict_bbox, pre_dict_label_index, scores

    def vis_bbox(self, rgb_img, bbox, label_index, scores, label_names):
        """
        用图像显示物体检测的预测结果的函数。

        Parameters
        ----------
        rgb_img:rgb图像  
            对象的图像数据
        bbox: list
            物体的BBox列表
        label_index: list
            对物体标签的索引
        scores: list
            物体的确信度
        label_names: list
            标签名的排列

        Returns
        -------
        没有。 显示在rgb_img中加入了物体检测结果的图像。  
        """

        # 边框颜色的设定
        num_classes = len(label_names)  # 班级数(背景)
        colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

        # 图像的显示
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_img)
        currentAxis = plt.gca()

        # BBox的循环  
        for i, bb in enumerate(bbox):

            # 标签名
            label_name = label_names[label_index[i]]
            color = colors[label_index[i]]  # クラスごとに別の色の枠を与える

            # 添加在框上的标签例:person;0.72
            if scores is not None:
                sc = scores[i]
                display_txt = '%s: %.2f' % (label_name, sc)
            else:
                display_txt = '%s: ans' % (label_name)

            # 框的坐标
            xy = (bb[0], bb[1])
            width = bb[2] - bb[0]
            height = bb[3] - bb[1]

            # 画长方形
            currentAxis.add_patch(plt.Rectangle(
                xy, width, height, fill=False, edgecolor=color, linewidth=2))

            # 在长方形框的左上角画标签
            currentAxis.text(xy[0], xy[1], display_txt, bbox={
                             'facecolor': color, 'alpha': 0.5})
