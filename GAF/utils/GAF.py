
from pyts.image import GramianAngularField
import csv
import numpy as np


def GAF(sin_data,image_size):

    sin_data = np.array(sin_data)
    sin_data = sin_data.reshape(1, -1)
    print(sin_data)
    gasf = GramianAngularField(image_size=image_size, method='summation', overlapping='False')
    sin_gasf = gasf.fit_transform(sin_data)
    print(sin_gasf)
    gadf = GramianAngularField(image_size=image_size, method='difference')
    sin_gadf = gadf.fit_transform(sin_data)
    images = [sin_gasf[0], sin_gadf[0]]
    with open('GAF1.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(images[0])
    return images
