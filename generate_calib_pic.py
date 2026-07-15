import numpy as np
import cv2
import arucoUtilities as au
import svgwrite
from svgwrite import cm, mm
import json
import argparse

class charuco2svg(object):
    def __init__(self,SQUARE_X, SQUARE_Y, SQUARE_LENGTH, MARKER_LENGTH, DICT_STRING, SVG_PATH='' ):
        #Variable Parsing
        self.SQUARE_X = SQUARE_X
        self.SQUARE_Y = SQUARE_Y
        self.SQUARE_LENGTH = SQUARE_LENGTH
        self.MARKER_LENGTH = MARKER_LENGTH
        self.DICT_STRING = DICT_STRING
        self_PATH=SVG_PATH
        #OpenCV objects
        self.charucoBoard = cv2.aruco.CharucoBoard(
            (SQUARE_X, SQUARE_Y), SQUARE_LENGTH, MARKER_LENGTH, au.toDict(DICT_STRING))
        self.DICT = au.toDict(DICT_STRING)
        #SVG related objects and variables 
        self.px_m = MARKER_LENGTH/au.markerWidth(DICT_STRING)
        self. markerOffset = ((SQUARE_LENGTH-MARKER_LENGTH)/2.0)
        self.drawing = svgwrite.Drawing(SVG_PATH, size=(SQUARE_X*SQUARE_LENGTH*100*cm, SQUARE_Y*SQUARE_LENGTH*100*cm), profile='full')


    def drawMarker(self,markerImage,point): 
        self.drawing.add(self.drawing.rect(insert=(point[0]*100*cm, point[1]*100*cm),
                size=(self.MARKER_LENGTH*100*cm, self.MARKER_LENGTH*100*cm), fill='black'))
        for y in range(markerImage.shape[0]):
            for x in range(markerImage.shape[1]):
                if markerImage[y][x] == 255:
                    self.drawing.add(self.drawing.rect(insert=((point[0]+x*self.px_m)*100*cm,(point[1]+ y*self.px_m)*100*cm), 
                                                        size=(self.px_m*100*cm, self.px_m*100*cm), fill='white'))

    def generateSVG(self):
        oddRows=self.SQUARE_Y%2
        markerPositions = [[oddRows==(i+j)%2 for i in range(self.SQUARE_X)]for j in range(self.SQUARE_Y)]
        markers = au.getMarkers(self.charucoBoard.getIds().flatten(),self.DICT,au.markerWidth(self.DICT_STRING))


        markerIdx = 0
        for y in range(self.SQUARE_Y):
            for x in range(self.SQUARE_X):
                if markerPositions[y][x]:
                    self.drawMarker(markers[markerIdx], (x*self.SQUARE_LENGTH+self.markerOffset, y*self.SQUARE_LENGTH+self.markerOffset))
                    markerIdx+=1
                else:
                    self.drawing.add(self.drawing.rect(insert=(x*self.SQUARE_LENGTH*100*cm, y*self.SQUARE_LENGTH*100*cm),
                                    size=(self.SQUARE_LENGTH*100*cm, self.SQUARE_LENGTH*100*cm), fill='black'))
                    
        self.drawing.save()

    def generatePNG(self, png_path, dpi=300, margin_px=0):
        # Calculate pixel size based on DPI (1 inch = 0.0254 meters)
        pixels_per_meter = dpi / 0.0254
        width_px = int(round(self.SQUARE_X * self.SQUARE_LENGTH * pixels_per_meter))
        height_px = int(round(self.SQUARE_Y * self.SQUARE_LENGTH * pixels_per_meter))
        
        out_size = (width_px + 2 * margin_px, height_px + 2 * margin_px)
        
        img = self.charucoBoard.generateImage(out_size, marginSize=margin_px)
        cv2.imwrite(png_path, img)


def generate_charuco_png(squares_x, squares_y, square_length, marker_length, dict_string, png_path, dpi=300, margin_px=0):
    """
    Generates a ChArUco board image and saves it to a PNG file.
    
    :param squares_x: Number of squares in X direction
    :param squares_y: Number of squares in Y direction
    :param square_length: Length of one square side (in meters)
    :param marker_length: Length of one marker side (in meters)
    :param dict_string: Name of the ArUco dictionary (e.g. 'DICT_4X4_50')
    :param png_path: Output path for the PNG file
    :param dpi: Dots Per Inch, used to determine the image resolution based on the physical size
    :param margin_px: Minimum margin size in pixels
    """
    dictionary = au.toDict(dict_string)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)
    
    # Calculate pixel size based on DPI (1 inch = 0.0254 meters)
    pixels_per_meter = dpi / 0.0254
    width_px = int(round(squares_x * square_length * pixels_per_meter))
    height_px = int(round(squares_y * square_length * pixels_per_meter))
    
    out_size = (width_px + 2 * margin_px, height_px + 2 * margin_px)
    
    img = board.generateImage(out_size, marginSize=margin_px)
    cv2.imwrite(png_path, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates a charucoBoard svg, png, and also a JSON file with the parameters")
    parser.add_argument('squaresX',help="squaresX parameter used for OpenCv charuco board initialization")
    parser.add_argument('squaresY',help="squaresY parameter used for OpenCv charuco board initialization")
    parser.add_argument('squareLength',help="squareLength parameter used for OpenCv charuco board initialization")
    parser.add_argument('markerLength',help="markerLength parameter used for OpenCv charuco board initialization")
    parser.add_argument('dictionary',help="dictionary parameter used for OpenCv charuco board initialization")
    parser.add_argument('--png', action='store_true', help="Also generate a PNG file")
    parser.add_argument('--dpi', type=int, default=300, help="DPI for the PNG image (default: 300)")
    parser.add_argument('--margin', type=int, default=0, help="Margin for the PNG image in pixels (default: 0)")

    args = parser.parse_args()

    svg_file_path = f"charucoBoard_{args.squaresX}x{args.squaresY}_{float(args.squareLength)*1000}_{float(args.markerLength)*1000}.svg"
    png_file_path = f"charucoBoard_{args.squaresX}x{args.squaresY}_{float(args.squareLength)*1000}_{float(args.markerLength)*1000}.png"
    json_file_path = f"charucoBoard_{args.squaresX}x{args.squaresY}_{float(args.squareLength)*1000}_{float(args.markerLength)*1000}.json"
    
    params = {'squaresX':args.squaresX,'squaresY':args.squaresY,'squareLength':args.squareLength,'markerLength':args.markerLength,'dictionary':args.dictionary}

    with open(json_file_path, 'w') as outfile:
        json.dump(params, outfile,indent=4)
    print("Wrote charuco board params to {}".format(json_file_path))
   
    charuco2svg(int(args.squaresX),int(args.squaresY),float(args.squareLength),float(args.markerLength),args.dictionary,svg_file_path).generateSVG()
    print("Saved charuco board as {}".format(svg_file_path))

    if args.png:
        generate_charuco_png(int(args.squaresX), int(args.squaresY), float(args.squareLength), float(args.markerLength), args.dictionary, png_file_path, dpi=args.dpi, margin_px=args.margin)
        print("Saved charuco board as {}".format(png_file_path))