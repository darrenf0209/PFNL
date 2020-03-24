''' This file deals with the pre-processing of images before they are fed for training or testing '''

import image_slicer
path = './test/vid4/calendar/truth/Frame 001.png'

num_tiles = 16
tiles = image_slicer.slice(path, num_tiles, save=False)
image_slicer.save_tiles(tiles, directory='./test/vid4/calendar/truth_tiles', prefix='tile_16', format='png')
print('Image has been sliced into {} tiles'.format(num_tiles))