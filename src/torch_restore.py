import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from FEMxML.utils_ml import plot_prection


class modelRestore():
    def __init__(self, savedPath, trainFlag=False, scalerPath=None):
        self.savedPath = savedPath
        self.modelPath = os.path.join(self.savedPath, 'entire_model.pt')
        self.trainFlag = trainFlag
        self.device = torch.device('cpu')

        # restore model
        self.model = self.load().to(self.device)
        self.startModel()

        modelPath = savedPath
        print()
        print('-'*80)
        print('Model restored from %s' % modelPath)
        if scalerPath:
            temp = scalerPath
        else:
            temp = self.savedPath

    def load(self):
        model = torch.load(self.modelPath, map_location=self.device)
        return model

    def startModel(self, ):
        if self.trainFlag:
            self.model.train()
        else:
            self.model.eval()

    def modelEvaluation(self, inputs, outputs):
        prediction = self.get_prediction(inputs=inputs)
        n = len(outputs)
        # plot the training model prediction
        plot_prection(y_origin=outputs, y_predict=prediction, root_path=self.savedPath)

    def get_prediction(self, inputs):
        inputsNorm = torch.tensor(inputs, dtype=torch.float)
        return self.model(inputsNorm).detach().numpy()

    def modelEvaluation_y_dy(self, x, y, dy):
        y_pre, dy_pre = self.get_prediction_y_dy(x=x)
        # plot the training model prediction
        plot_prection(y_origin=y, y_predict=y_pre, root_path=self.savedPath, s='y & y_pre')
        plot_prection(y_origin=dy, y_predict=dy_pre, root_path=self.savedPath, s='dy & dy_pre')

    def get_prediction_y_dy(self, x):
        xx = torch.tensor(x, dtype=torch.float, requires_grad=True).to(self.device)
        y = self.model(xx)
        dy = torch.autograd.grad(outputs=y, inputs=xx,
                            grad_outputs=torch.ones(y.size()).to(self.device),
                            retain_graph=True,
                            create_graph=True, only_inputs=True)[0]
        return y.detach().numpy(), dy.detach().numpy()
