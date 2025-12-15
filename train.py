

import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
import sys,time
from os.path import join
from lib.extract_patches import get_data_train
from lib.losses.loss import *
from lib.visualize import group_images, save_img
from lib.common import *
from lib.dataset import TrainDataset
from torch.utils.data import DataLoader
from config import parse_args
from lib.logger import Logger, Print_Logger
from collections import OrderedDict
from lib.metrics import Evaluate
import models
from test import Test


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#  Load the data and extract patches
def get_dataloader(args):              # 加载训练集和验证集数据
    patches_imgs_train, patches_masks_train = get_data_train(
        data_path_list = args.train_data_path_list,
        patch_height = args.train_patch_height,
        patch_width = args.train_patch_width,  #图像分块大小
        N_patches = args.N_patches,             #图像分块数量
        inside_FOV = args.inside_FOV #select the patches only inside the FOV  (default == False)
    )
    val_ind = random.sample(range(patches_masks_train.shape[0]),int(np.floor(args.val_ratio*patches_masks_train.shape[0]))) #随机取出两万个数字
    train_ind = set(range(patches_masks_train.shape[0])) - set(val_ind)
    train_ind = list(train_ind)

    train_set = TrainDataset(patches_imgs_train[train_ind,...],patches_masks_train[train_ind,...],mode="train")
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)

    val_set = TrainDataset(patches_imgs_train[val_ind,...],patches_masks_train[val_ind,...],mode="val")
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    # Save some samples of feeding to the neural network，保存50张样本图
    N_sample = min(patches_imgs_train.shape[0], 50)
    save_img(group_images((patches_imgs_train[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
              join(args.outf, args.save, "sample_input_imgs.png"))
    save_img(group_images((patches_masks_train[0:N_sample, :, :, :]*255).astype(np.uint8), 10),
              join(args.outf, args.save,"sample_input_masks.png"))
    return train_loader,val_loader

# train
def train(train_loader,net,criterion,optimizer,device):
    net.train()
    train_loss = AverageMeter()

    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader), total=len(train_loader)): #tqdm用于进度条可视化
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), inputs.size(0))
    log = OrderedDict([('train_loss',train_loss.avg)]) #记录epoch的损失值
    return log

# val
def val(val_loader,net,criterion,device):
    net.eval()
    val_loss = AverageMeter()     # 参数记录，更新，求和，求平均
    evaluater = Evaluate()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(val_loader), total=len(val_loader)):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss.update(loss.item(), inputs.size(0))

            outputs = outputs.data.cpu().numpy()
            targets = targets.data.cpu().numpy()
            evaluater.add_batch(targets,outputs[:,1])
            confusion, accuracy, specificity, sensitivity, precision = evaluater.confusion_matrix()
    log = OrderedDict([('val_loss', val_loss.avg),                       # 评价指标
                       ('val_acc', accuracy),
                       ('val_f1', evaluater.f1_score()),
                       ('val_auc_roc', evaluater.auc_roc()),
                       ('SE', sensitivity),
                       ('SP', specificity),
                       ('PR',precision),
                       ('MCC',evaluater.mcc())])
    return log

def main():
    setpu_seed(2022)   # 设计随机种子
    args = parse_args()    # 相关参数
    save_path = join(args.outf, args.save)
    save_args(args,save_path)  #保存参数信息

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")   # 调用GPU
    cudnn.benchmark = True  # 自动寻找最适合当前配置的高效算法，来达到优化运行效率的效果
    
    log = Logger(save_path)    # 储存日志
    sys.stdout = Print_Logger(os.path.join(save_path,'train_log.txt'))
    print('The computing device used is: ','GPU' if device.type=='cuda' else 'CPU')

    # net = models.UNetFamily.U_Net(1,2).to(device)
    #net = models.UNetFamily.U_Net(1,2).to(device)
    net=models.Test(2).to(device)

    # DataParallel
    #net = nn.DataParallel(net)

    print("Total number of parameters: " + str(count_parameters(net))) #计算模型参数量

    # log.save_graph(net,torch.randn((1,1,64,64)).to(device).to(device=device))  # Save the model structure to the tensorboard file，可用于打印模型结构
    # torch.nn.init.kaiming_normal(net, mode='fan_out')      # Modify default initialization method
    # net.apply(weight_init)

    # criterion = LossMulti(jaccard_weight=0,class_weights=np.array([0.5,0.5]))
    # NLLLoss损失函数
    # criterion = CrossEntropyLoss2d() # Initialize loss function
    criterion = CE_Dice_Loss()  # Initialize loss function

    # create a list of learning rate with epochs
    # lr_epoch = np.array([50, args.N_epochs])
    # lr_value = np.array([0.001, 0.0001])
    # lr_schedule = make_lr_schedule(lr_epoch,lr_value)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)
    # optimizer = optim.SGD(net.parameters(),lr=lr_schedule[0], momentum=0.9, weight_decay=5e-4, nesterov=True)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)  # 定义优化算法，学习率
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.N_epochs, eta_min=0) #定义学习率衰减函数，余弦退火调整学习率

    # The training speed of this task is fast, so pre training is not recommended
    if args.pre_trained is not None:
        # Load checkpoint.   checkpoint相当于保存模型的参数，优化器参数，loss，epoch的文件夹
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.outf + '%s/latest_model.pth' % args.pre_trained)
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch']+1
    train_loader, val_loader = get_dataloader(args)  # create dataloader

    if args.val_on_test:
        print('\033[0;32m===============Validation on Testset!!!===============\033[0m')
        val_tool = Test(args) #实例化

    best = {'epoch':0,'AUC_roc':0.5} # Initialize the best epoch and performance(AUC of ROC)
    trigger = 0  # Early stop Counter
    for epoch in range(args.start_epoch,args.N_epochs+1):
        print('\nEPOCH: %d/%d --(learn_rate:%.6f) | Time: %s' % \
            (epoch, args.N_epochs,optimizer.state_dict()['param_groups'][0]['lr'], time.asctime()))

        # train stage
        train_log = train(train_loader,net,criterion, optimizer,device)
        # val stage
        if not args.val_on_test:
            val_log = val(val_loader,net,criterion,device)
        else:
            val_tool.inference(net)
            val_log = val_tool.val()

        log.update(epoch,train_log,val_log) # Add log information
        lr_scheduler.step()

        # Save checkpoint of latest and best model.   # 保存检查点
        state = {'net': net.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['val_auc_roc'] > best['AUC_roc']:
            print('\033[0;33mSaving best model!\033[0m')
            torch.save(state, join(save_path, 'best_model.pth'))
            best['epoch'] = epoch
            best['AUC_roc'] = val_log['val_auc_roc']
            trigger = 0
        print('Best performance at Epoch: {} | AUC_roc: {}'.format(best['epoch'],best['AUC_roc']))
        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()
if __name__ == '__main__':
    main()
