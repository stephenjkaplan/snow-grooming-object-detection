"""
Data acquisition. If running the main notebook in Google Colaboratory, must move the resulting images to a folder
named "classes" in your Google Drive.
"""
from openimages.download import download_dataset


def download_google_open_images(class_labels, limit_per_class=5000, dest_dir='data'):
    """
    Download images and boundary box labels from https://storage.googleapis.com/openimages/web/index.html.

    :param class_labels:
    :param limit_per_class:
    :param dest_dir:
    :return:
    """
    download_dataset(dest_dir=dest_dir, class_labels=class_labels, annotation_format='darknet', limit=limit_per_class)


if __name__ == '__main__':
    download_google_open_images(class_labels=['Person', 'Snowmobile', 'Snowplow', 'Tree', 'Building',
                                              'Bench', 'Traffic sign', 'Street light'])
