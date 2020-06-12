import pandas as pd
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dropout
import keras
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers.core import Dense, Activation
import matplotlib.pyplot as plt

#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        # plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g-.', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            # plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k-,', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

learning_rate = 5e-3
dropout_rate = 0.05   # 0.6
patience = 2
batch_size = 20
epochs = 500
inputfile = 'F:\\work\\homework\\课程文件\\网课--疫情期间\\机器学习\\大作业\\数据集\\4饮酒与在校表现的关系\\code\\data\\por_to_num.csv' #灰色预测后保存的路径
inputfile2 = 'F:\\work\\homework\\课程文件\\网课--疫情期间\\机器学习\\大作业\\数据集\\4饮酒与在校表现的关系\\code\\data\\mat_to_num.csv' #灰色预测后保存的路径
outputfile = 'F:\\work\\homework\\课程文件\\网课--疫情期间\\机器学习\\大作业\\数据集\\4饮酒与在校表现的关系\\code\\data\\revenue.xls' #神经网络预测后保存的结果
modelfile = 'F:\\work\\homework\\课程文件\\网课--疫情期间\\机器学习\\大作业\\数据集\\4饮酒与在校表现的关系\\code\\data\\net.model' #模型保存路径

#读取数据
train_data = pd.read_csv(inputfile)
test_data = pd.read_csv(inputfile2) #读取数据

data_train = train_data.ix[0:650,:]   # [649,34]
print(data_train)
data_mean = data_train.mean()
data_std = data_train.std()
data_train = (data_train - data_mean)/data_std #数据标准化
train_data['refer'] = ((train_data['30']+train_data['31']+train_data['32'])/3).as_matrix()
print('train_data[refer]',train_data['refer'].shape)

data_test = test_data.ix[0:396,:]   # [395,34]
test_mean = data_test.mean()
test_std = data_test.std()
data_test = ((data_test - test_mean)/test_std).as_matrix() #数据标准化
test_data['refer'] = ((test_data['30']+test_data['31']+test_data['32'])/3).as_matrix()
print('test_data[refer]',test_data['refer'])


#建立模型
model = Sequential()
model.add(Dense(input_dim=34, output_dim=20))
model.add(Activation('relu'))
# model.add(Dropout(dropout_rate))
# model.add(Dense(input_dim=20, output_dim=10))
# model.add(Activation('relu'))
model.add(Dropout(dropout_rate))
model.add(Dense(input_dim=20, output_dim=1))
model.compile(loss='mean_squared_error', optimizer='adam') #编译模型

optimizer = Adam(lr=learning_rate)
model.compile(optimizer=optimizer,
              loss='mean_squared_error',
              weighted_metrics=['acc'])
history = LossHistory()
model.summary()
es_callback = EarlyStopping(monitor='val_loss ', patience=patience)   # val_weighted_acc
tb_callback = TensorBoard(batch_size=batch_size)
mc_callback = ModelCheckpoint('logs/best_model.h5',
                              monitor='val_weighted_acc',
                              save_best_only=True,
                              save_weights_only=True)

# 训练
validation_data = (data_test, test_data['refer'])
model.fit([data_train], train_data['refer'],
          epochs=epochs,
          batch_size=batch_size,
          validation_data=validation_data,
          shuffle=True,
          callbacks=[es_callback,tb_callback,mc_callback,history]  #
          )

prediction = model.predict(data_test)
print("prediction",prediction)
# 加载最好的模型
model.load_weights('logs/best_model.h5')

# Evaluate model
eval_results = model.evaluate([data_test], test_data['refer'],
                              batch_size=batch_size,
                              verbose=0)
loss = []
loss.append(eval_results[1])

print('Done.\n'
      'Test loss: {}\n'
      'Test accuracy: {}'.format(*eval_results))



# 绘制acc-loss曲线
history.loss_plot('epoch')