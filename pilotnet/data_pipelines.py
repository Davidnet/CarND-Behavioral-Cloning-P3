import os

import dicto as do

import numpy as np
import pandas as pd
import tensorflow as tf

CSV_FILENAME = "driving_log.csv"

def serving_input_fn(params):

    input_image = tf.placeholder(
        dtype=tf.float32,
        shape=[None, None, None, 3],
        name="input_image"
    )

    images = tf.image.resize_images(input_image, [params.image_height, params.image_width])


    crop_window = get_crop_window(params)
    images = tf.image.crop_to_bounding_box(images, *crop_window)

    images = tf.image.resize_images(images, [params.resize_height, params.resize_width])

    images = (images / 255.0) * 2.0 - 1.0

    return tf.estimator.export.ServingInputReceiver(
        features = dict(
            image=images,
        ),
        receiver_tensors = dict(
            image=input_image,
        ),
    )



def input_fn(data_dir, params):

    csv_filepath = os.path.join(data_dir, CSV_FILENAME)
    df = pd.read_csv(csv_filepath)
    df = normalize_dataframe(df)

    # There are whitespaces problems for column filename
    # df["filepath"] = df["filename"].str.strip()
    df["filepath"] = df["filename"].apply(lambda row: os.path.join(data_dir, row.strip()))

    df = process_dataframe(df, params)

    if params.only_center_camera:
        df = df[df.camera == 1]


    # Shuffle the dataset
    df = df.sample(frac=1)

    tensors = dict(
        filepath=df.filepath.values,
        steering=df.steering.values,
        camera=df.camera.values,
        original_steering=df.original_steering.values,
    )

    if "flipped" in df:
        tensors["flipped"] = df.flipped.values.astype(np.int32)

    ds = tf.data.Dataset.from_tensor_slices(tensors)

    ds = ds.shuffle(buffer_size=params.buffer_size, reshuffle_each_iteration=True)

    ds = ds.apply(tf.data.experimental.map_and_batch(
        lambda row: process_data(row, params),
        batch_size=params.batch_size,
        num_parallel_calls=params.n_threads,
        drop_remainder=True,
    ))

    # Please check dataset options
    ds = ds.prefetch(tf.contrib.data.AUTOTUNE)

    return ds

def normalize_dataframe(df):

    df_L = df.copy()
    df_C = df.copy()
    df_R = df.copy()

    df_L["camera"] = 0
    df_C["camera"] = 1
    df_R["camera"] = 2

    df_L["filename"] = df_L["left"]
    df_C["filename"] = df_C["center"]
    df_R["filename"] = df_R["right"]

    df_L = df_L.drop(["left", "center", "right"], axis=1)
    df_C = df_C.drop(["left", "center", "right"], axis=1)
    df_R = df_R.drop(["left", "center", "right"], axis=1)

    df = pd.concat([df_L, df_C, df_R])

    return df


def process_dataframe(df, params):

    df = df.copy()
    df_flipped = df.copy()

    df["flipped"] = False
    df_flipped["flipped"] = True

    df = pd.concat([df, df_flipped])
    df["original_steering"] = df.steering

    cam0 = df.camera == 0
    cam2 = df.camera == 2

    df.loc[cam0, "steering"] = df[cam0].steering + params.angle_correction
    df.loc[cam2, "steering"] = df[cam2].steering - params.angle_correction

    # flip
    flipped = df.flipped
    df.loc[flipped, "steering"] = -df[flipped].steering

    return df


def process_data(row, params):

    # Read image
    image = tf.read_file(row["filepath"])
    image = tf.image.decode_and_crop_jpeg(
        contents=image,
        crop_window=get_crop_window(params),
        channels=3
    )

    image = tf.image.resize_images(image, [params.resize_height, params.resize_width])

    image = tf.cond(
        row["flipped"] > 0,
        lambda: tf.image.flip_left_right(image),
        lambda: image,
    )


    if params.angle_noise_std > 0:
        noise = tf.random_normal([], mean=0.0, stddev=params.angle_noise_std)
        row["steering"] = tf.cast(row["steering"], tf.float32) + noise

    else:
        row["steering"] = tf.cast(row["steering"], tf.float32)

    image = (image / 255.0)* 2.0 - 1.0

    row["image"] = image

    return row

def get_crop_window(params):

    final_height = params.image_height - (params.crop_up + params.crop_down)
    final_width = params.image_width


    return [
        params.crop_up,
        0,
        final_height,
        final_width,
    ]

if __name__ == "__main__":
    module_path = os.path.dirname(__file__)
    configs_path = os.path.join(module_path, "configs", "train.yml")
    params = do.load(configs_path)