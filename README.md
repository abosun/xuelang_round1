###如果出现程序无法运行及其他相关问题，请及时联系我，万分感谢！  

姓名：李明阳
QQ：413033003  
Email：lmy413033003@126.com  
Tel：18392118194
##开发环境：
Python = 2.7.14  
Tensorflow = 1.5  
Numpy = 1.14.2  
CUDA = 8.0.44  
cudnn = 6  
proto = 3.4.0  
ubuntu = 16.04.4  

##目录结构：
--/
-- tocrop43yc.py # 生成异常样本数据
-- tocrop43zc.py # 生成正常样本数据
-- xuelang_round1_test_b # 测试图像文件夹  
-- round1 # 代码目录
-- data # 训练数据  
---- trainImg # 训练图像文件夹  
---- trainXml # 训练标签文件夹  
---- xuelang # 存储tfrecord及预处理图像
-- code # 代码目录
---- train_inception_v4.py # 训练主程序  
---- getTest_by_pb.py # 测试程序
---- inv4_640.pb # 已训练冻结模型
---- to_pb.py # 已冻结的训练模型
---- inception_v4.ckpt # 下载地址http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz
-- submit # 提交目录

##利用冻结数据进行测试
```
python code/getTest_by_pb.py

```
##数据集生成
```
sh preprocess.sh
```
##训练模型
```
# 训练模型
CUDA_VISIBLE_DEVICES=0,1 python code/train_inception_v4.py --train_dir=./code/train_640/ --dataset_name=image --dataset_split_name=train --dataset_dir=./data/xuelang/ --batch_size=10 --max_number_of_steps=200000  --model_name=inception_v4  --checkpoint_path=./code/inception_v4.ckpt --train_image_size=640 --num_readers=20 --num_clones=2
# 冻结模型
python code/to_pb.py --model_name=inception_v4 --checkpoint_path=./code/train_640/model.ckpt-20482
# 这一步需要指定checkpoint文件
```
