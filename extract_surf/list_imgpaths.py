import sys, os

"""
Feed me a directory containing images, and I'll spit out a text file,
where each row is the path to that image. 
This text file will then be the input to extract_surf_batch.py.
"""

def is_image_ext(filepath):
    IMG_EXTS = ('.jpg', '.png', 'tif', 'bmp', 'tiff')
    return os.path.splitext(filepath)[1].lower() in IMG_EXTS

def list_imgpaths(imgsdir, outname):
    outfile = open(outname, 'w')
    for dirpath, dirname, filenames in os.walk(imgsdir):
        for imgname in [f for f in filenames if is_image_ext(f)]:
            imgpath = os.path.join(dirpath, imgname)
            outfile.write(imgpath + '\n')
    outfile.close()

def main():
    args = sys.argv
    if len(args) <= 1:
        print 'Not enough inputs. Please give me a directory.'
        exit(1)
    else:
        imgsdir = args[1]

    if len(args) <= 2:
        print "You didn't provide an outfile file name. Going with \
the default: imgpaths_out.txt"
        outname = 'imgpaths_out.txt'
    else:
        outname = args[2]

    list_imgpaths(imgsdir, outname)
    
if __name__ == '__main__':
    main()
