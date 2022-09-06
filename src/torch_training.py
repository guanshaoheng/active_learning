import os
import time
import torch
import numpy as np
from FEMxML.utils_ml import splitTrainValidation, check_mkdir, writeDown
from utilSelf.general import echo


class train_class():
    def __init__(self, model, patienceNum, savePath, device, retrainFlag=False,
                 epoch_num=None, optimMode='adam', save_model_during_process_flag=False, verbose_flag=False):
        self.device = device
        self.model = model.to(torch.device(self.device))
        self.patienceNum = patienceNum
        self.savePath = savePath
        self.verbose_flag = verbose_flag
        self.save_model_during_process_flag = save_model_during_process_flag
        if 'BFGS' in optimMode:
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=0.05)
            self.checkStep = int(50)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), )
            self.checkStep = int(2500)
        self.loss = torch.nn.MSELoss()
        self.retrainFlag = retrainFlag
        self.epoch_num = epoch_num
        check_mkdir(self.savePath)

    def echo(self, samplesLen):
        info = '-' * 80 + '\n' + \
               '%s' % time.strftime('%Y/%m/%d  %H:%M:%S') + '\n' + \
               'PatienceNum:\t%d' % self.patienceNum + '\n' + \
               'Save path:\t%s' % self.savePath + '\n' + \
               'Number of training samples:\t%d' % samplesLen + '\n'

        if not self.retrainFlag:
            info += 'Model architecture:\t %s' % self.model + '\n' + \
               'Optimizer:\t%s' % self.optimizer + '\n' + \
                    '-' * 80 + '\n'
            self.writeDown(info)
        else:
            info+= '-' * 80 + '\n'
        print(info)
        return

    def writeDown(self, info):
        if self.retrainFlag:
            # print('-'*100)
            # print('Append the history file and retain!')
            f = open(os.path.join(self.savePath, 'history.dat'), 'a')
        else:
            # print('-'*100)
            # print('Delete the history file and begin training!')
            f = open(os.path.join(self.savePath, 'history.dat'), 'w')
            self.retrainFlag = True
        f.writelines(info)
        f.close()

    def save(self, savedPath, iter=None):
        if iter is not None:
            fname = 'entire_model_iter%d.pt' % iter
        else:
            fname = 'entire_model.pt'
        torch.save(self.model, os.path.join(savedPath, fname))

    def load(self, savedPath="../simu/ptModel"):
        model = torch.load(os.path.join(savedPath, 'entire_model.pt'))
        return model

    def loadCheckPoint(self, savePath="../simu/ptModel"):
        # model = Net()
        # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        checkpoint = torch.load(savePath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        self.model.eval()
        # - or -
        # model.train()

    def optimode_selection(self, optimMode):
        echo('the optimizer re-initialization, mode：%s' % optimMode)
        if optimMode and 'BFGS' in optimMode:
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=0.05)
            self.checkStep = int(50)
            self.patienceNum = int(100)
        elif optimMode and 'adam' in optimMode:
            self.optimizer = torch.optim.Adam(self.model.parameters(), )
            self.checkStep = int(1000)
            self.patienceNum = int(50000)
        elif optimMode and 'RMSprop' in optimMode:
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), )
            self.checkStep = int(2500)
            self.patienceNum = int(5000)
        elif optimMode and 'SGD' in optimMode:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
            self.checkStep = int(2500)
            self.patienceNum = int(5000)
        else: # default using the adama optimizer
            self.optimizer = torch.optim.Adam(self.model.parameters(), )
            self.checkStep = int(1000)
            self.patienceNum = int(50000)

    def train(self, inputs, outputs, epochMax=None, optimMode='adam', epoch_Num=None):
        self.optimode_selection(optimMode)
        self.echo(samplesLen=len(inputs))
        noProgressNum = 0
        epochNum = 0 if not epoch_Num else epoch_Num + 2
        miniLoss = 1e32

        # training data & validation data
        inputs = torch.from_numpy(inputs).float().to(self.device)
        outputs = torch.from_numpy(outputs).float().to(self.device)
        # inputs =
        # outputs = torch.tensor(outputs, dtype=torch.float, device=self.device)
        x_train, x_validation, y_train, y_validation = splitTrainValidation(inputs, outputs)

        start_time = time.time()
        while noProgressNum < self.patienceNum:
            epochNum += 1
            if epochMax and epochNum > epochMax:
                break
            noProgressNum += 1

            # forward + backward + optimize
            def closure():
                self.optimizer.zero_grad()
                outputsPredNorm = self.model(x_train)
                loss = self.loss(outputsPredNorm, y_train)
                # zero the parameter gradients
                loss.backward()
                return loss

            self.optimizer.step(closure)

            if epochNum % self.checkStep == 0 and epochNum != 0 and self.save_model_during_process_flag:
                # save the trained model along the training process
                tempPath = os.path.join(self.savePath, 'epoch_%d' % epochNum)
                check_mkdir(tempPath)
                self.save(savedPath=tempPath)

            # print statistics
            if epochNum % self.checkStep == 0 or epochNum == 1:  # print every 2000 mini-batches
                outputsNormTrain = self.model(x_train)
                lossCurrentTrain = self.loss(outputsNormTrain,
                                             y_train).item()
                # lossCurrent = self.loss(self.model(x_validation), y_validation).item()
                outputsNormValidation = self.model(x_validation)
                lossCurrent = self.loss(outputsNormValidation,
                                        y_validation).item()
                if lossCurrent < miniLoss:
                    noProgressNum = 0
                    ProgressFlag = True
                    miniLoss = lossCurrent
                    self.save(savedPath=self.savePath)
                else:
                    ProgressFlag = False
                info = 'Epoch:%d\t trainLoss: %e validationLoss: %e miniLoss: %e' % \
                       (epochNum, lossCurrentTrain, lossCurrent, miniLoss) + \
                       ('\tImproved!' if ProgressFlag else '\tNo improvement!') + \
                       "\tConsumedTime %e mins" % ((time.time() - start_time) / 60.)
                print(info)
                self.writeDown(info + '\n')
        print('No improvement in %d epochs, finished Training' % self.patienceNum)
        return epochNum


class train_dy_class(train_class):
    def __init__(self, model, patienceNum, savePath, device, retrainFlag=False,
                 epoch_num=None, optimMode='adam', dyWeight=0.1, save_model_during_process_flag=False,
                 num_batches=512, verbose_flag=False):
        train_class.__init__(
            self, model=model, patienceNum=patienceNum, savePath=savePath, device=device, retrainFlag=retrainFlag,
            epoch_num=epoch_num, optimMode=optimMode, save_model_during_process_flag=save_model_during_process_flag,
        verbose_flag=verbose_flag)
        self.dyWeight = dyWeight
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=800, gamma=0.9)
        self.num_batches= num_batches

    def trainOnce(self, x, y, dy):
        self.optimizer.zero_grad()
        yPrediction, dyPrediction = self.get_y_dy(x)
        # norm = torch.linalg.norm(dyPrediction_dx, dim=1)
        loss0 = self.loss(yPrediction, y)
        loss1 = self.loss(dyPrediction, dy)
        loss = loss0 + self.dyWeight * loss1
        loss.backward()
        self.optimizer.step()

    def get_y_dy(self, x):
        xx = torch.tensor(x, dtype=torch.float, requires_grad=True).to(self.device)
        y = self.model(xx)
        dy = torch.autograd.grad(outputs=y, inputs=xx,
                                 grad_outputs=torch.ones(y.size()).to(self.device),
                                 retain_graph=True,
                                 create_graph=True, only_inputs=True)[0]
        return y, dy

    def train(self, x, y, dy, epoch_max, optimMode=None):
        self.optimode_selection(optimMode)
        startTime = time.time()
        # network training
        echo('Training...')
        self.echo(samplesLen=len(x))

        x_train, x_val, y_train, y_val, dy_train, dy_val = splitTrainValidation(
            inputs=x, outputs=y, d_outouts=dy, valRatio=0.2)

        epoch = 0
        min_loss, trial_num = 1e32, 0
        while True:
            if self.num_batches == -1:
                # x_train.requires_grad= True
                self.trainOnce(
                    x=x_train,
                    y=torch.from_numpy(y_train).float().to(self.device),
                    dy=torch.from_numpy(dy_train).float().to(self.device))
            else:
                indices = np.arange(len(x_train))
                np.random.shuffle(indices)
                for batch in np.array_split(indices, self.num_batches):
                    self.trainOnce(
                        x=x_train[batch],
                    y=torch.from_numpy(y_train[batch]).float().to(self.device),
                    dy=torch.from_numpy(dy_train[batch]).float().to(self.device))

            if epoch % self.checkStep == 0:
                y_prediction_train, dy_prediction_train = self.get_y_dy(x_train)
                y_prediction_val, dy_prediction_val = self.get_y_dy(x_val)
                loss_train = (self.loss(y_prediction_train, torch.from_numpy(y_train).float().to(self.device))
                        +self.dyWeight*self.loss(dy_prediction_train, torch.from_numpy(dy_train).float().to(self.device))).cpu().item()
                loss_val = (self.loss(y_prediction_val, torch.from_numpy(y_val).float().to(self.device))
                        +self.dyWeight*self.loss(dy_prediction_val, torch.from_numpy(dy_val).float().to(self.device))).cpu().item()
                if loss_val < min_loss:
                    trial_num = 0
                    min_loss = loss_val
                    message = 'Improved!'
                    self.save(self.savePath)
                else:
                    trial_num += 1*self.checkStep
                    message = 'Noimproved! in %d/%d tirals' % (trial_num, self.patienceNum)

                info = "Epoch: %d \t lr: %.3e \t loss_train: %.3e Loss_val: %.3e \t miniLoss: %.3e \t %s \t timeConsumed: %.3e mins\n" % \
                       (epoch, self.optimizer.param_groups[0]['lr'], loss_train, loss_val, min_loss, message,
                        (time.time() - startTime) / 60.)
                print(info)
                writeDown(info, self.savePath, appendFlag=True)

                if epoch >= epoch_max:
                    info = '\n' + '-' * 80 + '\n' + \
                           'Training Ended till epoch: %d >= epochMax %d' % (epoch, epoch_max) + '\n' + info
                    print(info)
                    writeDown(info, self.savePath, appendFlag=True)
                    break
                if trial_num > self.patienceNum:
                    info = '\n' + '-' * 80 + '\n' + \
                           'Training Ended till trial_num: %d >= patienceNum %d' % (trial_num, self.patienceNum) + '\n' + info
                    print(info)
                    writeDown(info, self.savePath, appendFlag=True)
                    break
            if epoch % (self.checkStep * 100) == 0 and self.save_model_during_process_flag:
                saveDir = os.path.join(self.savePath, 'epoch_%d' % (epoch + 1))
                check_mkdir(saveDir)
                self.save(self.savePath)
            self.scheduler.step()  # used for the learning rate decay
            epoch += 1
