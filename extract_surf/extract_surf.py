#!/usr/bin/env python

import cv

def parse():
    import argparse

    parser = argparse.ArgumentParser(description = "SURF feature extraction.")
    parser.add_argument('imagepath', type = str, metavar = 'ImagePath', help = 'input image path')
    parser.add_argument('-s', '--show', dest = 'isShowImg', type = int, choices = [0, 1], default = 0, help = '1: show image, 0: not show image')
    parser.add_argument('-w', '--write', dest = 'isWriteFile', type = int, choices = [0, 1], default = 0, help = '1: write features to file, 0: not write to file')
    parser.add_argument('-o', '--outfile', dest = 'out_path', type = str, default = '', help = 'output file path')
    
    return parser.parse_args()
    
def read_file_folder(file_path):
    idx_slash = file_path.rfind('/')
    if idx_slash == -1:
        file_path = './'+file_path
        idx_slash = file_path.rfind('/')
    idx_dot = file_path.rfind('.', idx_slash)
    if idx_dot == -1:
        idx_dot = len(file_path)
    
    return (file_path[idx_slash+1:idx_dot], file_path[:idx_slash])

def extract_surf(image_path, isShowImg = 0, isWriteFile = 0, out_path = ''):
    im = cv.LoadImageM(image_path, cv.CV_LOAD_IMAGE_GRAYSCALE)        
    (keypoints, descriptors) = cv.ExtractSURF(im, None, cv.CreateMemStorage(), (1, 400, 3, 4))
    print 'extracted {0}, number of features = {1}'.format(image_path, len(keypoints))
    
    if isShowImg:
        im_rgb = cv.LoadImageM(image_path, cv.CV_LOAD_IMAGE_COLOR)
        for ((x, y), laplacian, size, dir, hessian) in keypoints:
            print 'x={0:d} y={1:d} laplacian={2} size={3} dir={4} hessian={5}'.format(int(x), int(y), laplacian, size, dir, hessian)
            cv.Circle(im_rgb, (int(x), int(y)), 3, cv.Scalar(255,0,0), 1);
        
        cv.ShowImage('Image', im_rgb)
        cv.WaitKey(0)
    
    if isWriteFile:
        if out_path == '':
            (out_name, out_path) = read_file_folder(image_path)
        else:
            (out_name, tmp_path) = read_file_folder(image_path)
    
        if out_path[-1] != '/':
            out_path = out_path + '/'
    
        outfile = open(out_path + out_name +'_surf.txt', 'w')
        outfile.write('{0}   {1}\n'.format(len(keypoints), 128))
    
        i = 0
        for ((x, y), laplacian, size, dir, hessian) in keypoints:
            outfile.write('{0} {1} {2} {3} {4} {5} \n'.format(x, y, laplacian, size, dir, hessian))
            for feat in descriptors[i]:
                outfile.write('{0} '.format(feat))
            outfile.write('\n')
            i = i+1
    
    return (keypoints, descriptors)

if __name__ == '__main__':
    args = parse()
    extract_surf(args.imagepath, args.isShowImg, args.isWriteFile, args.out_path)
    