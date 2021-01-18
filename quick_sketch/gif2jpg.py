"""Converts png, bmp and gif to jpg, removes descriptions and resizes the image to a maximum of 1920x1080."""

from PIL import Image
from glob import glob
import PIL
import sys
import os


src_dir = os.path.join('data')


def compress_image(image, infile):
    name = infile.split('.')
    outfile = os.path.join(src_dir, name[0] + '.jpg')
    image.save(outfile)


def processImage():
    listing = os.listdir(src_dir)
    for infile in listing:
        img = Image.open(os.path.join(src_dir, infile))

        if img.format == "JPEG":
            image = img.convert('RGB')
            compress_image(image, infile)
            img.close()

        elif img.format == "GIF":
            i = img.convert("RGBA")
            bg = Image.new("RGBA", i.size)
            image = Image.composite(i, bg, i)
            compress_image(image, infile)
            img.close()

        elif img.format == "PNG":
            try:
                image = Image.new("RGB", img.size, (255,255,255))
                image.paste(img,img)
                compress_image(image, infile)
            except ValueError:
                image = img.convert('RGB')
                compress_image(image, infile)
            img.close()

        elif img.format == "BMP":
            image = img.convert('RGB')
            compress_image(image, infile)
            img.close()

if __name__ == '__main__':
    processImage()
