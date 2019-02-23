import os
import pandas as pd
import shutil
import dicto as do

from imgaug import augmenters as iaa
from dask import dataframe as dd
from dask.diagnostics import ProgressBar
from PIL import Image
from fire import Fire

from .utils import load_image as load_single_image

@do.fire_options("/code/pilotnet/configs/data-augmentation.yml")
def main(raw_dir, augmented_dir, rm, params):
    """
    Args:
        rm: bool
    """
    params = do.Dicto(params)

    if rm and os.path.exists(augmented_dir):
        shutil.rmtree(augmented_dir)

    csv_filepath = os.path.join(raw_dir, "driving_log.csv")
    # df = pd.read_csv(raw_dir)
    df = pd.read_csv(csv_filepath)

    if not "filepath" in df.columns:
        for path_column in ["center", "left", "right"]:
            # There are whitespace characters in the csv so that's why we strip characters.
            df[path_column + "_filepath"] = df[path_column].apply(lambda x: os.path.join(raw_dir, x.strip()))

        df["folder"] = raw_dir

    sd = augment_dataset(df, params, save_dir=augmented_dir, return_image=False)

    print("Augmenting images")

    with ProgressBar():
        # df = sd.compute(get=get)
        df = sd.compute(scheduler='processes')

    print("Saving CSVs")


    for _, dfg in df.groupby(["folder", "augment_idx"]):

        sample = dfg.iloc[0]
        folder = sample.folder + "_" + str(sample.augment_idx)
        csv_path = os.path.join(augmented_dir, folder, "driving_log.csv")

        del dfg["left_image"]
        del dfg["left_filepath"]
        del dfg["center_image"]
        del dfg["center_filepath"]
        del dfg["right_image"]
        del dfg["right_filepath"]

        del dfg["augment_idx"]
        del dfg["folder"]
        del dfg["augmented"]

        dfg.to_csv(csv_path, index = False)
 


def get_seq(params):

    """
    Main filters and augmentations for pilotnet data augmentation.
    """
    filters = iaa.SomeOf(params.filters_repeat, [
        iaa.ChangeColorspace("BGR"),
        iaa.ChangeColorspace("GRAY"),
        iaa.GaussianBlur(sigma=(0.0, 3.0)),
        iaa.AverageBlur(k=(2, 9)),
        iaa.MedianBlur(k=(3, 9)),
        iaa.Add((-40, 40), per_channel=0.5),
        iaa.Add((-40, 40)),
        iaa.AdditiveGaussianNoise(scale=0.05*255, per_channel=0.5),
        iaa.AdditiveGaussianNoise(scale=0.05*255),
        iaa.Multiply((0.5, 1.5), per_channel=0.5),
        iaa.Multiply((0.5, 1.5)),
        iaa.MultiplyElementwise((0.5, 1.5)),
        iaa.ContrastNormalization((0.5, 1.5)),
        iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
        iaa.ElasticTransformation(alpha=(0, 2.5), sigma=0.25),
        iaa.Sharpen(alpha=(0.6, 1.0)),
        iaa.Emboss(alpha=(0.0, 0.5)),
        iaa.CoarseDropout(0.2, size_percent=0.00001, per_channel = 1.0),
    ])
    affine = iaa.Affine(
        rotate=(-7, 7),
        scale=(0.9, 1.1),
        translate_percent=dict(x = (-0.05, 0.05)),
        mode = "symmetric",
    )

    return iaa.Sequential([
        filters,
        affine,
    ])

def load_image(row, path_column, name_column, seq, save_dir = None, return_image = True):

    image = load_single_image(row[path_column])

    if row["augmented"]:
        try:
            image = seq.augment_image(image)
        except:
            print(row)
            raise

    
    if save_dir is not None:
        folder = row.folder + "_" + str(row.augment_idx)
        image_folder = os.path.join(save_dir, folder, "IMG")
        # image_path = os.path.join(image_folder, row[name_column].strip())

        image_filepath = os.path.basename(row[name_column])
        image_path = os.path.join(image_folder, image_filepath)
        
        os.makedirs(image_folder, exist_ok=True)

        im = Image.fromarray(image)
        im.save(image_path)
    
    if return_image:
        return (image,)
    else:
        return None
 
def augment_dataset(df, params, save_dir=None, return_image=True):
    """
    Main loop for augment dataset.
    """

    seq = get_seq(params)


    df = df.copy()

    df["augmented"] = False

    df["augment_idx"] = 0

    dfs = [df]

    for i in range(params.augmentation_factor - 1):
        dfi = df.copy()
        dfi["augmented"] = True
        dfi["augment_idx"] = i + 1
        dfs.append(dfi)

    df = pd.concat(dfs)

    sd = dd.from_pandas(df, npartitions=params.n_threads)

    # import pdb; pdb.set_trace()
    if "filepath" in sd.columns:
        sd["image"] = sd.apply(lambda row: load_image(row, "filepath", "filename", seq, save_dir=save_dir, return_image=return_image), axis = 1, meta=tuple)
    else:
        sd["left_image"] = sd.apply(lambda row: load_image(row, "left_filepath", "left", seq, save_dir=save_dir, return_image=return_image), axis = 1, meta=tuple)
        sd["center_image"] = sd.apply(lambda row: load_image(row, "center_filepath", "center", seq, save_dir=save_dir, return_image=return_image), axis = 1, meta=tuple)
        sd["right_image"] = sd.apply(lambda row: load_image(row, "right_filepath", "right", seq, save_dir=save_dir, return_image=return_image), axis = 1, meta=tuple)

    return sd

if __name__ == "__main__":
    Fire(main)