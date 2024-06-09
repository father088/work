## 所需环境
torch  
torchvision  
tensorboard  
torchsummary  
scipy  
numpy  
matplotlib  
opencv_python  
tqdm  
Pillow  
h5py

## 环境配置
```python
pip install -r requirements.txt
```

## 文件下载
训练所需的权重可在百度网盘中下载。  
链接: https://pan.baidu.com/s/1gPDsDVX1lbcSNqCKsvzz0A   
提取码: 3mjs   

VOC数据集下载地址如下，里面已经包括了训练集、测试集、验证集（与测试集一样），无需再次划分：  
链接: https://pan.baidu.com/s/19Mw2u_df_nBzsC2lg20fQA    
提取码: j5ge   

## 训练步骤
### a、训练VOC07数据集
1. 数据集的准备   
**本文使用VOC格式进行训练，训练前需要下载好VOC07的数据集，解压后放在根目录**  

2. 数据集的处理   
运行voc_annotation.py生成根目录下的2007_train.txt和2007_val.txt。   

3. 开始网络训练   
train.py的默认参数用于训练VOC数据集，直接运行train.py即可开始训练。   

4. 训练结果预测   
训练结果预测需要用到两个文件，分别是yolo.py和predict.py。我们首先需要去yolo.py里面修改model_path以及classes_path，这两个参数必须要修改。   
**model_path指向训练好的权重文件，在logs文件夹里。   
classes_path指向检测类别所对应的txt。**   
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。   

### b、训练自己的数据集
1. 数据集的准备  
**本文使用VOC格式进行训练，训练前需要自己制作好数据集，**    
训练前将标签文件放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。   
训练前将图片文件放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。   
训练前将训练集的txt文件放在VOCdevkit文件夹下的VOC2007文件夹下的ImageSets/Main中。
训练前将测试集的txt文件放在VOCdevkit文件夹下的VOC2007文件夹下的ImageSets/Main中。
数据集的格式如下：  
VOCdevkit  
--VOC2007  
----Annotations  
------xml files (eg. 000001.xml)  
----JPEGImages  
------image files (eg. 000001.jpg)  
----ImageSets  
------Main  
--------train.txt  
--------test.txt  
--------trainval.txt  

2. 数据集的处理  
在完成数据集的摆放之后，我们需要利用voc_annotation.py,获得训练用的2007_train.txt和2007_val.txt。   
修改voc_annotation.py里面的参数。第一次训练可以仅修改classes_path，classes_path用于指向检测类别所对应的txt。   
训练自己的数据集时，可以自己建立一个cls_classes.txt，里面写自己所需要区分的类别。   
model_data/cls_classes.txt文件内容为：      
```python
aeroplane
bicycle
bird
...
```
修改voc_annotation.py中的classes_path，使其对应cls_classes.txt，并运行voc_annotation.py。  

3. 开始网络训练  
**训练的参数较多，均在train.py中，其中最重要的部分是train.py里的classes_path。**  
**classes_path用于指向检测类别所对应的txt，这个txt和voc_annotation.py里面的txt一样！**  
修改完classes_path后就可以运行train.py开始训练了，在训练多个epoch后，权重会生成在logs文件夹中。  

4. 训练结果预测  
训练结果预测需要用到两个文件，分别是根目录下的yolo.py和predict.py。在yolo.py里面修改model_path以及classes_path。  
**model_path指向训练好的权重文件，在logs文件夹里。  
classes_path指向检测类别所对应的txt。**  
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。  

## 预测步骤
### a、使用预训练权重
1. 在百度网盘下载权重文件，放入model_data，运行predict.py，输入图片路径，例如：
```python
img/street.jpg
```
2. 在predict.py里面进行设置可以进行fps测试和video视频检测。  
### b、使用自己训练的权重
1. 按照训练步骤训练。  
2. 在yolo.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权重文件，classes_path是model_path对应分的类**。  

3. 测试单张图片：设置predict.py里的mode为predict，运行predict.py，输入图片路径，例如：
    ```python
    img/street.jpg
    ```
4. 测试多张图片：设置predict.py里的mode为dir_predict，修改dir_origin_path参数为预测的图片文件夹路径，运行predict.py。
5. 测试fps：设置predict.py里的mode为fps，运行predict.py。

## 评估步骤 
### a、评估VOC07的测试集
1. 本文使用VOC格式进行评估。VOC07已经划分好了测试集，无需利用voc_annotation.py生成ImageSets文件夹下的txt。
2. 在yolo.py里面修改model_path以及classes_path。**model_path指向训练好的权重文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**  
3. 运行test.py即可获得评估结果，评估结果会保存在map_out文件夹中。

### b、评估自己的数据集
1. 本文使用VOC格式进行评估。  
2. 如果在训练前已经运行过voc_annotation.py文件，代码会自动将数据集划分成训练集、验证集和测试集。如果想要修改测试集的比例，可以修改voc_annotation.py文件下的trainval_percent。trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1。train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1。
3. 利用voc_annotation.py划分测试集后，前往test.py文件修改classes_path，classes_path用于指向检测类别所对应的txt，这个txt和训练时的txt一样。评估自己的数据集必须要修改。
4. 在根目录下的yolo.py里面修改model_path以及classes_path。**model_path指向训练好的权重文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**  
5. 运行test.py即可获得评估结果，评估结果会保存在map_out文件夹中。

