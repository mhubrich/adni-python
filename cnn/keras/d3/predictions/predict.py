from cnn.keras.models.d3G_pos.model import build_model
from cnn.keras.prediction.predict_generator import predict_generator
from cnn.keras.d3.preprocessing.image_processing import inputs
from cnn.keras.utils.write_prediction import write_prediction
from utils.split_scans import read_imageID


classes = ['Normal', 'AD']
target_size = (29, 29, 29)
batch_size = 16
load_all_scans = False
path_weights = '/home/mhubrich/weights.14-loss_0.401-acc_0.816.h5'
output_name = 'G_intnorm_pos_11_14-11_47_4.csv'

interval = range(11, 78 - target_size[0], 4)
grid = [(x, y, z) for x in interval for y in interval for z in interval]


def predict():
    # Get inputs for labeling
    scans_test = read_imageID('/home/mhubrich/ADNI_intnorm_npy', '/home/mhubrich/val_intnorm')
    test_inputs = inputs(scans_test, target_size, batch_size, load_all_scans, classes, 'predict', seed=grid)

    # Set up the model
    model = build_model(num_classes=len(classes), input_shape=(1,)+target_size)
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.load_weights(path_weights)

    # Start labeling
    pred, filenames = predict_generator(model,
                                        test_inputs,
                                        val_samples=test_inputs.nb_sample,
                                        max_q_size=16,
                                        nb_preprocessing_threads=1)

    return pred, filenames


if __name__ == "__main__":
    predictions, filenames = predict()
    write_prediction(output_name, predictions, filenames)

