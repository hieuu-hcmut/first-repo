import tensorflow as tf


class KAIST_DATASET:
    def __init__(self, batchsize, height, width, train_file, test_file):
        self.buffer_size = 1000
        self.batchsize = batchsize
        self.height = height
        self.width = width
        self.train_file = train_file
        self.test_file = test_file

    @staticmethod
    def load(image_file):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_jpeg(image)
        loaded_image = tf.cast(image, tf.float32)

        return loaded_image

    def resize(self, img):
        resized_image = tf.image.resize(img, [self.height, self.width])
        return resized_image

    @staticmethod
    def normalize(resized_image):
        normalized_image = (resized_image / 127.5) - 1
        return normalized_image

    def process(self, lwir_path):
        loaded_lwir_image = self.load(lwir_path)
        resized_lwir_image = self.resize(loaded_lwir_image)
        normalized_lwir_image = self.normalize(resized_lwir_image)

        visible_path = tf.strings.regex_replace(lwir_path, "lwir", "visible")
        loaded_visible_image = self.load(visible_path)
        resized_visible_image = self.resize(loaded_visible_image)
        normalized_visible_image = self.normalize(resized_visible_image)

        return normalized_lwir_image, normalized_visible_image

    def export_data(self):
        train_ds =  tf.data.TextLineDataset([self.train_file])
        test_ds = tf.data.TextLineDataset([self.test_file])

        train_ds = train_ds.shuffle(self.buffer_size).map(self.process, num_parallel_calls=tf.data.AUTOTUNE).batch(self.batchsize).prefetch(1)
        test_ds = test_ds.map(self.process, num_parallel_calls=tf.data.AUTOTUNE).batch(self.batchsize)
        return train_ds, test_ds