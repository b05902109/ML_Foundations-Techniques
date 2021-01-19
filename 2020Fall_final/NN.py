import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import argparse, os
import numpy as np
from math import sqrt
from dataset import TrainDataset, TestDataset
from util import progress_bar
from model import Net

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', default='./data', type=str, help='Data folder')
    parser.add_argument('-ckpt', default='./ckpt', type=str, help='Model checkpoint folder')
    parser.add_argument('-output', default='./predict.csv', type=str, help='Predict answers')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--is_canceled', action='store_true')
    parser.add_argument('--adr', action='store_true')


    arg = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[LOG] device: ', device)

    net = None

    if arg.train:
        if arg.is_canceled:
            net = Net(is_canceled=True).to(device)
        elif arg.adr:
            net = Net(adr=True).to(device)
        criterion1 = nn.CrossEntropyLoss()
        # criterion1 = nn.MSELoss()
        criterion2 = nn.MSELoss()
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        optimizer = optim.Adam(net.parameters(), lr=0.001)

        trainDataset = TrainDataset(arg.input)
        validData_df, validY, validDataReservation = trainDataset.readTrainData()
        validLabels = trainDataset.readTrainLabelData()
        validDataset = TrainDataset('')
        validDataset.initial(validData_df, validY, validDataReservation, validLabels)
        # trainDataset.oneHot()

        trainDataset.oneHot()
        validDataset.oneHot()
        trainDataset.toTensor()
        validDataset.toTensor()

        trainloader = torch.utils.data.DataLoader(trainDataset, batch_size=32, shuffle=True, num_workers=2)
        validloader = torch.utils.data.DataLoader(validDataset, batch_size=100, shuffle=False, num_workers=2)

        acc, err = 0, 10000000
        epochs = 20

        # training
        for epoch in range(epochs):
            net.train()
            train_loss1, train_loss2 = 0, 0
            train_acc1, train_err2 = 0, 0
            total = 0
            for batch_idx, (X, Y1, Y2) in enumerate(trainloader):
                # (X, Y1, Y2) == (torch.Size([128, 24]), torch.Size([128]), torch.Size([128]))
                X = Variable(X,requires_grad=True)
                # Y1 = Variable(Y1.int(),requires_grad=True)
                Y2 = Variable(Y2,requires_grad=True)
                X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)
                optimizer.zero_grad()
                # Y1_hat, Y2_hat = net(X)
                if arg.is_canceled:
                    Y1_hat = net(X)
                    loss1 = criterion1(Y1_hat, Y1.detach())
                    loss1.backward()
                    optimizer.step()
                    train_loss1 += loss1.item()
                    _, predicted = Y1_hat.max(1)
                    train_acc1 += predicted.eq(Y1).sum().item()

                elif arg.adr:
                    Y2_hat = net(X).squeeze()
                    loss2 = criterion2(Y2_hat, Y2.detach())
                    loss2.backward()
                    optimizer.step()
                    train_loss2 += loss2.item()
                    train_err2 += (Y2_hat - Y2).pow(2).sum().item()

                # elif arg.is_canceled and arg.adr:
                #     loss1 = criterion1(Y1_hat, Y1.detach())
                #     loss2 = criterion2(Y2_hat, Y2.detach())
                #     loss = loss1 + loss2
                #     loss.backward()


                total += Y1.size(0)
                # break
                progress_bar(batch_idx, len(trainloader), 'Loss1: %.3f | Loss2: %.3f | Acc1: %.3f | Err2: %.3f'
                             % (train_loss1/(batch_idx+1), train_loss2/(batch_idx+1), train_acc1/total, sqrt(train_err2/total)))

            # validation
            # continue
            # if epoch % 5 == 0:
            net.eval()
            with torch.no_grad():
                valid_loss1, valid_loss2 = 0, 0
                valid_acc1, valid_err2 = 0, 0
                total = 0
                is_canceled, adr = np.array([]), np.array([])
                for batch_idx, (X, Y1, Y2) in enumerate(validloader):
                    # X = Variable(X,requires_grad=True)
                    # Y1 = Variable(Y1,requires_grad=True)
                    # Y2 = Variable(Y2,requires_grad=True)
                    X, Y1, Y2 = X.to(device), Y1.to(device), Y2.to(device)

                    if arg.is_canceled:
                        Y1_hat = net(X)
                        loss1 = criterion1(Y1_hat, Y1.detach())
                        valid_loss1 += loss1.item()
                        _, predicted = Y1_hat.max(1)
                        valid_acc1 += predicted.eq(Y1).sum().item()
                        is_canceled = np.append(is_canceled, predicted.cpu().numpy())

                    elif arg.adr:
                        Y2_hat = net(X).squeeze()
                        loss2 = criterion2(Y2_hat, Y2.detach())
                        valid_loss2 += loss2.item()
                        valid_err2 += (Y2_hat - Y2).pow(2).sum().item()
                        Y2_hat = Y2_hat.cpu().numpy()
                        adr = np.append(adr, Y2_hat)

                    total += Y1.size(0)                    

                    progress_bar(batch_idx, len(validloader), 'Loss1: %.3f | Loss2: %.3f | Acc1: %.3f | Err2: %.3f'
                                 % (valid_loss1/(batch_idx+1), valid_loss2/(batch_idx+1), valid_acc1/total, sqrt(valid_err2/total)))

                # valid_err = validDataset.getLabelsAcc(is_canceled, adr)

                if arg.is_canceled:
                    if valid_acc1/total > acc:
                        state = {
                            'net': net.state_dict(),
                        }
                        if not os.path.isdir(arg.ckpt):
                            os.mkdir(arg.ckpt)
                        torch.save(state, os.path.join(arg.ckpt, 'best_is_canceled.pth'))
                        acc = valid_acc1/total
                        print('epoch:', epoch, ', valid_acc1:', valid_acc1/total)
                elif arg.adr:
                    if sqrt(valid_err2/total) < err:
                        state = {
                            'net': net.state_dict(),
                        }
                        if not os.path.isdir(arg.ckpt):
                            os.mkdir(arg.ckpt)
                        torch.save(state, os.path.join(arg.ckpt, 'best_adr.pth'))
                        err = sqrt(valid_err2/total)
                        print('epoch:', epoch, ', valid_err2:', sqrt(valid_err2/total))


    # testing
    if arg.test:
        testDataset = TestDataset(arg.input)
        testDataset.readTestData()
        testDataset.readTestLabelData()
        testDataset.oneHot()
        testDataset.toTensor()
        testloader = torch.utils.data.DataLoader(testDataset, batch_size=100, shuffle=False, num_workers=2)

        labels = None
        is_canceled, adr = np.array([]), np.array([])

        net = Net(is_canceled=True).to(device)
        state = torch.load(os.path.join(arg.ckpt, 'best_is_canceled.pth'))
        net.load_state_dict(state['net'])  
        net.eval()
        with torch.no_grad():
            for batch_idx, (X) in enumerate(testloader):
                X = X.to(device)
                Y1_hat = net(X)

                _, predicted = Y1_hat.max(1)
                is_canceled = np.append(is_canceled, predicted.cpu().numpy())

        net = Net(adr=True).to(device)
        state = torch.load(os.path.join(arg.ckpt, 'best_adr.pth'))
        net.load_state_dict(state['net'])  
        net.eval()
        with torch.no_grad():
            for batch_idx, (X) in enumerate(testloader):
                X = X.to(device)
                Y2_hat = net(X)

                Y2_hat = Y2_hat.cpu().numpy()
                adr = np.append(adr, Y2_hat)
        
        labels = testDataset.predictLabels(is_canceled, adr)
        labels.to_csv(arg.output, index=False)

# 1. no one hot
# 2. one hot
# 3. one hot, is_canceled adr train alone

# [============================ 2574/2574 =========================>]  Step: 3ms | Tot: 9s764ms | Loss1: 0.422 | Loss2: 0.000 | Acc1: 0.809 | Err2: 0.000
# [============================ 92/92 =============================>]  Step: 1ms | Tot: 218ms | Loss1: 0.449 | Loss2: 0.000 | Acc1: 0.791 | Err2: 0.000
# epoch: 0 , valid_acc1: 0.7912976932327539
# [============================ 2574/2574 =========================>]  Step: 4ms | Tot: 9s916ms | Loss1: 0.375 | Loss2: 0.000 | Acc1: 0.834 | Err2: 0.000
# [============================ 92/92 =============================>]  Step: 2ms | Tot: 251ms | Loss1: 0.438 | Loss2: 0.000 | Acc1: 0.789 | Err2: 0.000
# [============================ 2574/2574 =========================>]  Step: 6ms | Tot: 10s983ms | Loss1: 0.368 | Loss2: 0.000 | Acc1: 0.836 | Err2: 0.000
# [============================ 92/92 =============================>]  Step: 3ms | Tot: 222ms | Loss1: 0.433 | Loss2: 0.000 | Acc1: 0.796 | Err2: 0.000
# epoch: 2 , valid_acc1: 0.7961080135563573
# [============================ 2574/2574 =========================>]  Step: 3ms | Tot: 11s50ms | Loss1: 0.361 | Loss2: 0.000 | Acc1: 0.839 | Err2: 0.000
# [============================ 92/92 =============================>]  Step: 1ms | Tot: 248ms | Loss1: 0.434 | Loss2: 0.000 | Acc1: 0.795 | Err2: 0.000

# [============================ 2574/2574 =========================>]  Step: 14ms | Tot: 35s761ms | Loss1: 0.000 | Loss2: 1782.245 | Acc1: 0.000 | Err2: 42.218
# [1189/1875] [============================ 92/92 =============================>]  Step: 2ms | Tot: 993ms | Loss1: 0.000 | Loss2: 516.948 | Acc1: 0.000 | Err2: 22.774
# epoch: 0 , valid_err2: 22.77449254083792
# [============================ 2574/2574 =========================>]  Step: 16ms | Tot: 35s478ms | Loss1: 0.000 | Loss2: 1236.075 | Acc1: 0.000 | Err2: 35.158
# [============================ 92/92 =============================>]  Step: 13ms | Tot: 1s3ms | Loss1: 0.000 | Loss2: 591.366 | Acc1: 0.000 | Err2: 24.338
# [============================ 2574/2574 =========================>]  Step: 14ms | Tot: 31s637ms | Loss1: 0.000 | Loss2: 1079.708 | Acc1: 0.000 | Err2: 32.860
# [============================ 92/92 =============================>]  Step: 1ms | Tot: 871ms | Loss1: 0.000 | Loss2: 454.905 | Acc1: 0.000 | Err2: 21.353
# epoch: 2 , valid_err2: 21.35315658466837
# [============================ 2574/2574 =========================>]  Step: 14ms | Tot: 33s243ms | Loss1: 0.000 | Loss2: 1026.201 | Acc1: 0.000 | Err2: 32.035
# [============================ 92/92 =============================>]  Step: 11ms | Tot: 905ms | Loss1: 0.000 | Loss2: 452.733 | Acc1: 0.000 | Err2: 21.317
# epoch: 3 , valid_err2: 21.31724332256436
# [============================ 2574/2574 =========================>]  Step: 14ms | Tot: 32s357ms | Loss1: 0.000 | Loss2: 1003.164 | Acc1: 0.000 | Err2: 31.673
# [============================ 92/92 =============================>]  Step: 1ms | Tot: 877ms | Loss1: 0.000 | Loss2: 528.242 | Acc1: 0.000 | Err2: 23.017  
# [============================ 2574/2574 =========================>]  Step: 14ms | Tot: 31s215ms | Loss1: 0.000 | Loss2: 991.937 | Acc1: 0.000 | Err2: 31.496
# [============================ 92/92 =============================>]  Step: 1ms | Tot: 963ms | Loss1: 0.000 | Loss2: 410.490 | Acc1: 0.000 | Err2: 20.292 
# epoch: 5 , valid_err2: 20.291706307981503  
# [============================ 2574/2574 =========================>]  Step: 14ms | Tot: 34s754ms | Loss1: 0.000 | Loss2: 972.515 | Acc1: 0.000 | Err2: 31.185
# [============================ 92/92 =============================>]  Step: 11ms | Tot: 974ms | Loss1: 0.000 | Loss2: 422.726 | Acc1: 0.000 | Err2: 20.593
# [============================ 2574/2574 =========================>]  Step: 14ms | Tot: 34s635ms | Loss1: 0.000 | Loss2: 959.302 | Acc1: 0.000 | Err2: 30.973
# [============================ 92/92 =============================>]  Step: 11ms | Tot: 985ms | Loss1: 0.000 | Loss2: 457.075 | Acc1: 0.000 | Err2: 21.410
# [============================ 2574/2574 =========================>]  Step: 14ms | Tot: 34s562ms | Loss1: 0.000 | Loss2: 947.417 | Acc1: 0.000 | Err2: 30.781
# [============================ 92/92 =============================>]  Step: 11ms | Tot: 983ms | Loss1: 0.000 | Loss2: 411.149 | Acc1: 0.000 | Err2: 20.300
# [============================ 2574/2574 =========================>]  Step: 14ms | Tot: 34s933ms | Loss1: 0.000 | Loss2: 935.028 | Acc1: 0.000 | Err2: 30.579
# [============================ 92/92 =============================>]  Step: 11ms | Tot: 1s15ms | Loss1: 0.000 | Loss2: 441.470 | Acc1: 0.000 | Err2: 21.022
# [============================ 2574/2574 =========================>]  Step: 14ms | Tot: 35s11ms | Loss1: 0.000 | Loss2: 922.018 | Acc1: 0.000 | Err2: 30.365
# [============================ 92/92 =============================>]  Step: 11ms | Tot: 1s32ms | Loss1: 0.000 | Loss2: 578.200 | Acc1: 0.000 | Err2: 24.083
# [============================ 2574/2574 =========================>]  Step: 14ms | Tot: 35s59ms | Loss1: 0.000 | Loss2: 912.656 | Acc1: 0.000 | Err2: 30.211
# [============================ 92/92 =============================>]  Step: 11ms | Tot: 981ms | Loss1: 0.000 | Loss2: 422.199 | Acc1: 0.000 | Err2: 20.568
# [============================ 2574/2574 =========================>]  Step: 14ms | Tot: 34s789ms | Loss1: 0.000 | Loss2: 908.583 | Acc1: 0.000 | Err2: 30.143
# [============================ 92/92 =============================>]  Step: 9ms | Tot: 1s18ms | Loss1: 0.000 | Loss2: 438.053 | Acc1: 0.000 | Err2: 20.933
# [============================ 2574/2574 =========================>]  Step: 14ms | Tot: 34s889ms | Loss1: 0.000 | Loss2: 898.984 | Acc1: 0.000 | Err2: 29.983
# [============================ 92/92 =============================>]  Step: 11ms | Tot: 1s6ms | Loss1: 0.000 | Loss2: 440.785 | Acc1: 0.000 | Err2: 21.013
# [============================ 2574/2574 =========================>]  Step: 14ms | Tot: 34s841ms | Loss1: 0.000 | Loss2: 887.192 | Acc1: 0.000 | Err2: 29.786
# [============================ 92/92 =============================>]  Step: 11ms | Tot: 991ms | Loss1: 0.000 | Loss2: 446.959 | Acc1: 0.000 | Err2: 21.152
# [============================ 2574/2574 =========================>]  Step: 14ms | Tot: 34s927ms | Loss1: 0.000 | Loss2: 883.800 | Acc1: 0.000 | Err2: 29.729
# [============================ 92/92 =============================>]  Step: 11ms | Tot: 992ms | Loss1: 0.000 | Loss2: 421.097 | Acc1: 0.000 | Err2: 20.537
# [============================ 2574/2574 =========================>]  Step: 14ms | Tot: 34s977ms | Loss1: 0.000 | Loss2: 869.534 | Acc1: 0.000 | Err2: 29.488
# [============================ 92/92 =============================>]  Step: 11ms | Tot: 1s7ms | Loss1: 0.000 | Loss2: 462.332 | Acc1: 0.000 | Err2: 21.518
# [============================ 2574/2574 =========================>]  Step: 14ms | Tot: 34s920ms | Loss1: 0.000 | Loss2: 860.733 | Acc1: 0.000 | Err2: 29.338
# [============================ 92/92 =============================>]  Step: 11ms | Tot: 954ms | Loss1: 0.000 | Loss2: 410.198 | Acc1: 0.000 | Err2: 20.263
# epoch: 17 , valid_err2: 20.26328518111112
# [============================ 2574/2574 =========================>]  Step: 14ms | Tot: 32s897ms | Loss1: 0.000 | Loss2: 852.198 | Acc1: 0.000 | Err2: 29.193
# [============================ 92/92 =============================>]  Step: 11ms | Tot: 868ms | Loss1: 0.000 | Loss2: 407.461 | Acc1: 0.000 | Err2: 20.206
# epoch: 18 , valid_err2: 20.20601968268946
# [============================ 2574/2574 =========================>]  Step: 14ms | Tot: 32s206ms | Loss1: 0.000 | Loss2: 838.795 | Acc1: 0.000 | Err2: 28.962
# [============================ 92/92 =============================>]  Step: 11ms | Tot: 880ms | Loss1: 0.000 | Loss2: 405.610 | Acc1: 0.000 | Err2: 20.146
# epoch: 19 , valid_err2: 20.146420112705485
# [============================ 2574/2574 =========================>]  Step: 14ms | Tot: 32s223ms | Loss1: 0.000 | Loss2: 833.972 | Acc1: 0.000 | Err2: 28.879
# [============================ 92/92 =============================>]  Step: 11ms | Tot: 871ms | Loss1: 0.000 | Loss2: 469.748 | Acc1: 0.000 | Err2: 21.673
# [============================ 2574/2574 =========================>]  Step: 3ms | Tot: 32s377ms | Loss1: 0.000 | Loss2: 829.217 | Acc1: 0.000 | Err2: 28.797
# [============================ 92/92 =============================>]  Step: 11ms | Tot: 894ms | Loss1: 0.000 | Loss2: 410.318 | Acc1: 0.000 | Err2: 20.286
# [============================ 2574/2574 =========================>]  Step: 14ms | Tot: 32s820ms | Loss1: 0.000 | Loss2: 820.952 | Acc1: 0.000 | Err2: 28.652
# [============================ 92/92 =============================>]  Step: 11ms | Tot: 985ms | Loss1: 0.000 | Loss2: 401.735 | Acc1: 0.000 | Err2: 20.054
# epoch: 22 , valid_err2: 20.053654685073468