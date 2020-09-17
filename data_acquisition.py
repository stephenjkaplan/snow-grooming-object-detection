"""
Data acquisition. If running the main notebook in Google Colaboratory, must move the resulting images to a folder
named "classes" in your Google Drive.
"""
import os
from openimages.download import download_dataset


def download_google_open_images(class_labels, limit_per_class=5000, dest_dir='data'):
    """
    Download images and boundary box labels from https://storage.googleapis.com/openimages/web/index.html.

    Will create a folder for each object class, each containing a folder for the images and a folder for the .xml files
    (containing boundary box coordinates). Corresponding images and xml files will have the same file name (with
    different file extensions).

    :param list class_labels: List of valid classes from the Google Open Images dataset.
    :param int limit_per_class: Maximum number of images with annotations that will be downloaded per object class.
                                Defaults to 5000.
    :param str dest_dir: Directory to download files into.
    """
    # download data
    class_labels_cap = [o.capitalize() for o in class_labels]
    download_dataset(dest_dir=dest_dir, class_labels=class_labels_cap, annotation_format='pascal', limit=limit_per_class)

    # rename folders 'pascal' to 'xml_files' to be compatible with Snow_Grooming_Object Detection_with_PyTorch.ipynb
    for label in class_labels:
        os.rename(os.path.join(dest_dir, label, 'pascal'), os.path.join(dest_dir, label, 'xml_files'))


if __name__ == '__main__':
    download_google_open_images(class_labels=['Person', 'Snowmobile', 'Snowplow', 'Tree', 'Building',
                                              'Bench', 'Traffic sign', 'Street light', 'Truck'])
