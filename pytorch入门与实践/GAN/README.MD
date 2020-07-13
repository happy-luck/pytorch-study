## 文件说明

- `checkpoints/`： 用于保存训练好的模型，可使程序在异常退出后仍能重新载入模型，恢复训练
- `data/`：训练数据
- `imgs/`：保存生成图片
- `models.py`：模型定义
- `main.py`：主文件，训练和测试程序的入口，可通过不同的命令来指定不同的操作和参数
- `visualize.py`：可视化工具visdom的封装
- `README.md`：提供程序的必要说明

## 环境准备

- 本程序需要安装[PyTorch](https://pytorch.org/)
- 还需要通过`pip install -r requirements.txt` 安装其它依赖

## 数据准备

- 可以自己写爬虫爬取[Danbooru](http://link.zhihu.com/?target=http%3A//safebooru.donmai.us/)或者[konachan](http://konachan.net/)
- 如果你不想从头开始爬图片，可以直接使用爬好的头像数据（275M，约5万多张图片）：https://pan.baidu.com/s/1eSifHcA 提取码：g5qa
感谢知乎用户[何之源](https://www.zhihu.com/people/he-zhi-yuan-16)爬取的数据。
请把所有的图片保存于data/face/目录下，形如
```
data/
└── faces/
    ├── 0000fdee4208b8b7e12074c920bc6166-0.jpg
```

## 用法
如果想要使用visdom可视化，请先运行`python -m visdom.server`启动visdom服务
基本用法：
```
Usage： python main.py FUNCTION --key=value,--key2=value2 ..
```

- 训练
```bash
python main.py train --nogpu --visdom=False
```

- 生成图片
```bash
python main.py generate --nogpu --visdom=False \
            --netd-path=checkpoints/netd_200.pth \
            --netg-path=checkpoints/netg_200.pth \
            --gen-img=result.png \
            --gen-num=64
```
完整的选项及默认值
```python
    data_path = 'data/' # 数据集存放路径
    num_workers = 4 # 多进程加载数据所用的进程数
    image_size = 96 # 图片尺寸
    batch_size = 256
    max_epoch =  200
    lr1 = 2e-4 # 生成器的学习率
    lr2 = 2e-4 # 判别器的学习率
    beta1=0.5 # Adam优化器的beta1参数
    gpu=True # 是否使用GPU --nogpu或者--gpu=False不使用gpu
    nz=100 # 噪声维度
    ngf = 64 # 生成器feature map数
    ndf = 64 # 判别器feature map数
    
    save_path = 'imgs/' #训练时生成图片保存路径
    
    vis = True # 是否使用visdom可视化
    env = 'GAN' # visdom的env
    plot_every = 20 # 每间隔20 batch，visdom画图一次

    debug_file='/tmp/debuggan' # 存在该文件则进入debug模式
    d_every=1 # 每1个batch训练一次判别器
    g_every=5 # 每5个batch训练一次生成器
    decay_every=10 # 没10个epoch保存一次模型
    netd_path = 'checkpoints/netd_211.pth' #预训练模型
    netg_path = 'checkpoints/netg_211.pth'
    
    # 只测试不训练
    gen_img = 'result.png'
    # 从512张生成的图片中保存最好的64张
    gen_num = 64 
    gen_search_num = 512 
    gen_mean = 0 # 噪声的均值
    gen_std = 1 #噪声的方差
   
```
