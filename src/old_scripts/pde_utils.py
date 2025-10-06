import numpy as np
import warnings

# CREATED AS FUNCTION OF DATASET CLASS
# def cpoint_generation(minX, maxX, minY, maxY, dtm,
#                       mode = "even",
#                       num_lon_point = 100,
#                       num_lat_point = 100):
    
#     if mode == "even":
        
#         # create one-dimensional arrays for x and y
#         x = np.linspace(minX, maxX, num_lon_point)
#         y = np.linspace(minY, maxY, num_lat_point)[::-1]
#         # create the mesh based on these arrays
#         X, Y = np.meshgrid(x, y)
#         coords = np.stack([Y, X], axis = -1)
        
        
#         dtm_xy = dtm.sel(x = x, y = y,
#                         method = "nearest").values
        
#         coords = np.concat([coords, np.moveaxis(dtm_xy, 0, -1)], axis=-1)
#         coords = coords.reshape(coords.shape[0]*coords.shape[1], coords.shape[2])
        
#     elif mode == "urandom":
        
#         if(num_lon_point != num_lat_point):
#             warnings.warn("number of lat cpoints not equal to lon cpoints... the min is considered in the following")
#         num_cpoints = min(num_lon_point, num_lat_point)
        
#         x = np.random.uniform(low=minX, high=maxX, size=num_cpoints)
#         y = np.random.uniform(low=minY, high=maxY, size=num_cpoints)
        
#         dtm_xy = np.array([dtm.sel(x = x[i], y = y[i],
#                         method = "nearest").values for i in range(num_cpoints)])
        
#         coords = np.concat([np.expand_dims(y, 1), np.expand_dims(x, 1), dtm_xy], axis=-1)
        
#     return coords