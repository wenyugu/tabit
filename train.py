import torch
import argparse

from data.dataset import *
import torch.optim as optim
from tqdm import tqdm
from torch.optim import lr_scheduler
from time import time
from models.model import *
from models.metric import *
import sys
import os

def display_config(args):
    print('-------SETTINGS_________')
    for arg in vars(args):
        print("%15s: %s" % (str(arg), str(getattr(args, arg))))
    print('')

ROOT_PATH = os.path.abspath(os.path.dirname(__file__))

parser = argparse.ArgumentParser()
# basic setting
parser.add_argument('--model', type=str, default='cnn', help="model name")
parser.add_argument('--log_name', type=str, default='cnn', help="log file name")
parser.add_argument('--dataset', type=str, default='guitarset', help="dataset name")
parser.add_argument('--data_dir', type=str, default=ROOT_PATH + "/data/spec_repr/", help="dataset path")
parser.add_argument('--partition_file', type=str, default=ROOT_PATH + "/data/id.csv", help="parition file path")
parser.add_argument('--batchsize', type=int, default=128, help='training batch size')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs')
parser.add_argument('--save_every', type=int, default=5, help='frequency of saving model')
parser.add_argument('--num_workers', type=int, default=0, help='number of threads for data loader')
parser.add_argument('--log_path', type=str ,default='./result/log/', help="log path")
parser.add_argument('--save_model_path', type=str, default='./result/weight', help='location to save checkpoint models')
# parser.add_argument('--pretrained_path', type=str, default='pretrained/RRN-10L.pth', help="path of the pretrained model for evaluating")
parser.add_argument('--verbose', action='store_true', default=False, help="print extra process logs")

# training
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate.')
parser.add_argument('--decay', default=0.1, type=float,help="learning rate decay")

# testing
# parser.add_argument('--test', action='store_true', default=False, help="whether is testing or not")

args = parser.parse_args()

def main():
    start_time = time()
    # gpu_devices = os.environ['CUDA_VISIBLE_DEVICES']
    # gpu_devices = gpu_devices.split(',')
    # print("Using GPU", gpu_devices)

    if not torch.cuda.is_available():
        print('GPU not available')

    # cudnn.benchmark = True
    display_config(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def criterion(output, target):
        y = torch.max(target, 2)[1]
        z = -F.log_softmax(output, 2)
        z = z.gather(2, y.unsqueeze(2))
        return torch.sum(z, 1).mean()

    partition_csv = args.partition_file
    best_fscore = 0

    for k in range(6):
        model = Net()
        # model = model.cuda()
        # if len(gpu_devices) > 1:
        #     model = torch.nn.DataParallel(model)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        train_partition = []
        test_partition = []
        for item in list(pd.read_csv(partition_csv, header=None)[0]):
            fold = int(item.split("_")[0])
            if fold == k:
                test_partition.append(item)
            else:
                train_partition.append(item)

        train_dataset = GuitarSetDataset(train_partition)
        train_loader = DataLoader(dataset=train_dataset,
                                batch_size=args.batchsize,
                                shuffle=True,
                                pin_memory=torch.cuda.is_available(),
                                num_workers=args.num_workers)

        test_dataset = GuitarSetDataset(test_partition)
        test_loader = DataLoader(dataset=test_dataset,
                                batch_size=args.batchsize,
                                shuffle=False,
                                pin_memory=torch.cuda.is_available(),
                                num_workers=args.num_workers)

        model.train()
        for epoch in range(args.epoch):
            print('Epoch {}/{}'.format(epoch, args.epoch - 1))
            print('-' * 10)

            running_loss = 0.0

            for i, (inputs, labels) in tqdm(enumerate(train_loader)) if args.verbose else enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):

                    outputs = torch.reshape(model(inputs), (-1, 6, 21))
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            scheduler.step()

            epoch_loss = running_loss / len(train_dataset)

            print('Loss: {:.4f}'.format(epoch_loss))

            if (epoch + 1) % args.save_every == 0:
                checkpoint(model, epoch)
            # sys.stdout.flush()

        # testing
        precision = 0
        recall = 0
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in tqdm(enumerate(test_loader)) if args.verbose else enumerate(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = torch.reshape(model(inputs), (-1, 6, 21))

                tab_pred = F.one_hot(torch.max(F.softmax(outputs, 2), 2)[1], num_classes=21)[:, :, 1:]
                tab_gt = labels[:, :, 1:]
                tp = torch.sum(tab_pred * tab_gt)
                total_pred = torch.sum(tab_pred)
                true_pred = torch.sum(tab_gt)
                # precision recall f-score
                precision += tp / total_pred
                recall += tp / true_pred

        avg_precision = precision/ len(test_loader)
        avg_recall = recall / len(test_loader)
        avg_fscore = f_score(avg_precision, avg_recall)
        print('[Fold {}] Precision: {:5f} Recall: {:5f} F-score: {:5f}'.format(k, avg_precision, avg_recall, avg_fscore))
        if avg_fscore > best_fscore:
            best_fscore = avg_fscore
            torch.save(model.state_dict(), os.path.join(os.path.join(args.save_model_path, args.log_name), 'best_cnn.pth'))

        

    finish_time = time()
    print("Total Time Consumption:", (finish_time - start_time)/60, "min")
    return


def checkpoint(model, epoch):
    save_model_path = os.path.join(args.save_model_path, args.log_name)
    isExists = os.path.exists(save_model_path)
    if not isExists:
        os.makedirs(save_model_path)
    model_name = '{}'.format(args.model)+'_epoch_{}.pth'.format(epoch)
    torch.save(model.state_dict(), os.path.join(save_model_path, model_name))
    print('Checkpoint saved to {}'.format(os.path.join(save_model_path, model_name)))


if __name__ == '__main__':
    main()