import os
# Ensure tensorstore does not attempt to use GCE credentials
os.environ['GCE_METADATA_ROOT'] = 'metadata.google.internal.invalid'

import tensorstore as ts
import numpy as np
import tifffile

#Volumetric cutouts and point lookups for both the EM data and the segmentations can be performed using TensorStore
context = ts.Context({'cache_pool': {'total_bytes_limit': 1000000000}})

em_4nm = ts.open({
    'driver': 'neuroglancer_precomputed',
    'kvstore': {'driver': 'gcs', 'bucket': 'h01-release'},
    'path': 'data/20210601/4nm_raw'},
    read=True, context=context).result()[ts.d['channel'][0]]


# Coordinates for Vessel 1
x_start = 249800
x_size = 2200
y_start = 175000
y_size = 2200
range_start = 600
range_stop = 1519

## Coordinates for Vessel 2
#x_start = 207300
#x_size = 1800
#y_start = 161130
#y_size = 1800
#range_start = 0
#range_stop = 999

## Coordinates for Vessel 3
#x_start = 329300
#x_size = 1400
#y_start = 187700
#y_size = 1650
#range_start = 500
#range_stop = 1529

for zdepth in range(range_start,range_stop):
    img_cutout_4nm = em_4nm[x_start*2:(x_start+x_size)*2,y_start*2:(y_start+y_size)*2, zdepth].read().result()


    em_filename = f"image{zdepth:05d}z_" + str(x_start) + "x_" + str(y_start) + "y.tiff"
    tifffile.imwrite(em_filename, data=np.transpose(img_cutout_4nm))

