import numpy as np
import cv2
import os
from tqdm.auto import tqdm


all_files = os.listdir()

r_files = list(filter(lambda f: f.startswith('r-'), all_files))
d_files = list(filter(lambda f: f.startswith('d-'), all_files))

def get_time(f):
    assert f.startswith('r-') or f.startswith('d-')
    return float(f.removeprefix('r-').removeprefix('d-')
                  .removesuffix('.ppm').removesuffix('.pgm')
                  .replace('-',''))

times = [(get_time(f), f) for f in all_files if f.startswith('r-') or f.startswith('d-')]
times.sort()



for i in tqdm(range(len(times))):
    time, file = times[i]
    if file.startswith('d-'):
        continue

    time_d, file_d = next(filter(lambda f: f[1].startswith('d-'), times[i+1:]))

    image = cv2.imread(file, -1)
    depth = cv2.imread(file_d, -1)

    if abs(time - time_d) > 0.1:
        continue

    os.makedirs(f'data/frame_{i}', exist_ok=True)
    cv2.imwrite(f'data/frame_{i}/image.png', image)
    np.save(f'data/frame_{i}/image_depth.npy', depth)

