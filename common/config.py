#config.py
import configparser as parser
import os, sys

config = parser.ConfigParser()

if getattr(sys, 'frozen', False):
    # PyInstaller
    config_path = os.path.dirname(os.path.abspath(sys.executable))
    print('PyInstaller Packaged File')
    # print('config_path: ' + config_path)
    if config.read(config_path+'\config.ini'):
        print('parser: find config')
    else:
        print('parser: cannot find config')      
else:
    # Bianry
    config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
    print('Binary File')
    if config.read(config_path):
        print('parser: binary find config')
    else:
        print('parser: binary cannot find config')


CONFIG_IMAGE = config['IMAGE_PROPERTIES']
IMAGE_FILE_EXTESION = CONFIG_IMAGE['FILE_EXTENSION']
IMAGE_RESOLUTION = CONFIG_IMAGE['IMAGE_RESOLUTION']
IMAGE_UPSCALE_AFTER_PROGRESS = CONFIG_IMAGE['UPSCALE_AFTER_PROGRESS']
IMAGE_UPSCALE_AFTER_PROGRESS_RESOLUTION = CONFIG_IMAGE['UPSCALE_AFTER_PROGRESS_RESOLUTION']
IMAGE_EQUALIZE = CONFIG_IMAGE['IMAGE_EQUALIZE']

CONFIG_INTERPOLATE = config['INTERPLATE_PROPERTIES']
INTERPOLATE_STEPS = CONFIG_INTERPOLATE['INTERPOLATE_STEPS']
INTERPOLATE_WIDTH = CONFIG_INTERPOLATE['INTERPOLATE_WIDTH']
INTERPOLATE_DIAGONAL = CONFIG_INTERPOLATE['INTERPOLATE_DIAGONAL']

CONFIG_IO = config['IO_PROPERTIES']
IO_SOURCE_FOLDER = CONFIG_IO['SOURCE_FOLDER']
IO_FILE_OUTPUT_NAME = CONFIG_IO['FILE_OUTPUT_NAME']