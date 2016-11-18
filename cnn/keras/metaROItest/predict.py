from cnn.keras.prediction.predict_generator import predict_generator
from cnn.keras.utils.write_prediction import write_prediction
from cnn.keras.metaROItest.model import build_model
from cnn.keras.metaROItest.image_processing import inputs
from utils.split_scans import read_imageID


# Training specific parameters
classes = ['Normal', 'AD']
batch_size = 128
load_all_scans = True
# Paths
path_ADNI = '/home/mhubrich/ADNI_intnorm_metaROI1'
path_weights = None
output_name = ''


def predict():
    scans_val = read_imageID(path_ADNI, '/home/mhubrich/val_intnorm')
    val_inputs = inputs(scans_val, None, batch_size, load_all_scans, classes, 'predict', None)

    # Set up the model
    model = build_model(num_classes=len(classes))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.load_weights(path_weights)

    pred, filenames = predict_generator(model,
                                        test_inputs,
                                        val_samples=val_inputs.nb_sample,
                                        max_q_size=128,
                                        nb_preprocessing_threads=1)

    return pred, filenames


if __name__ == "__main__":
    predictions, filenames = predict()
    write_prediction(output_name, predictions, filenames)

