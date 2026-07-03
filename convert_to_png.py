#!/usr/bin/env python3
import os
import sys
import argparse
try:
    from PIL import Image
except ImportError:
    print("Error: The 'Pillow' library is required to run this script.", file=sys.stderr)
    print("Please install it using: pip install Pillow", file=sys.stderr)
    sys.exit(1)

def convert_to_png(input_path, output_path=None):
    """
    Converts an input image file to PNG format.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)
        
    if not output_path:
        # Generate output path by replacing the extension with .png
        base_name, _ = os.path.splitext(input_path)
        output_path = f"{base_name}.png"
        
    try:
        print(f"Opening image: {input_path}")
        with Image.open(input_path) as img:
            # Convert palette/grayscale/CMYK images to RGBA/RGB for safety when converting.
            if img.mode in ('CMYK', 'P'):
                if img.mode == 'CMYK':
                    img = img.convert('RGB')
            
            print(f"Saving as PNG: {output_path}")
            img.save(output_path, 'PNG')
            print("Conversion completed successfully!")
            
    except Exception as e:
        print(f"Error converting image: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Convert an image file to PNG format.")
    parser.add_argument("input_image", help="Path to the input image file (e.g., image.jpg, image.bmp, image.webp)")
    parser.add_argument("-o", "--output", help="Optional path for the output PNG file. Defaults to input filename with .png extension.")
    
    args = parser.parse_args()
    
    convert_to_png(args.input_image, args.output)

if __name__ == "__main__":
    main()
