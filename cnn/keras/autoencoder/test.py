import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

from model import build_model

target_size = (44, 52, 44)


model = build_model(input_shape=(1,)+target_size)
model.compile(loss='mse', optimizer='adadelta', metrics=['accuracy'])
model.load_weights('/home/markus/weights.492-loss_0.000-acc_0.650.h5')

path = '/home/markus/nswADNI_036_S_1001_PT_Co-registered,_Averaged_Br_20081121132951804_1_S59602_I127910.npy'
scan = np.load(path)
#scan = scan[47:, :, :] \
#           [:, 57:, :] \
#           [:, :, 47:]
scan = scan[23:67, :, :] \
           [:, 29:81, :] \
           [:, :, 23:67]

x = np.expand_dims(scan, axis=0)
x = np.expand_dims(x, axis=0)
pred = model.predict_on_batch(x)
pred = np.squeeze(pred)

k = 39
plt.figure(1)
plt.axis('off')
plt.imshow(pred[:,:,k])
plt.figure(2)
plt.axis('off')
plt.imshow(scan[:,:,k])
plt.show()

layer_output = K.function([model.layers[0].input], [model.layers[3].output])
pool = layer_output([x])[0]
pool = np.squeeze(pool)

plt.figure(1)
plt.axis('off')
for i in range(pool.shape[0]):
    plt.subplot(8,8,i+1)
    plt.imshow(pool[i, :, :, 9], cmap='gray')
    plt.axis('off')

plt.show()
