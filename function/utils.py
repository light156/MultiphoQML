import time
import os


def get_folder(dir, prefix=None, suffix=None):

    folder = time.strftime('%m%d_%H%M%S', time.localtime(time.time()))
    if prefix is not None:
        folder = prefix+'_'+folder

    if suffix is not None:
        folder = folder+'_'+suffix
    
    folder = os.path.join(dir, folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
        os.makedirs(folder+'/weights')
        
    return folder
