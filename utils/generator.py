# Image data generator for batch training
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def get_generators(base_dir = "./data/stage1", batch_size=1, input_shape=(300, 300, 300), valid_dir=None):
    #training generator
    train_datagen = ImageDataGenerator(
            rescale=1.,horizontal_flip=True)
    train_generator = train_datagen.flow_from_directory(
            base_dir,
            classes=['cancer', 'non_cancer'],
            target_size=input_shape,
            batch_size=batch_size,
            class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    #validation generator
    if valid_dir is not None:
        valid_datagen = ImageDataGenerator(rescale=1.)
        valid_generator = valid_datagen.flow_from_directory(
            valid_dir,
            classes=['cancer', 'non_cancer'],
            target_size=input_shape,
            batch_size=batch_size,
            class_mode='binary')

        return train_generator, valid_generator

    return train_generator

