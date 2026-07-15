#!/bin/bash

DICTIONARY="DICT_4X4_250"
DPI=300
MARGIN=0

# # Help message
# show_help() {
#   echo "Usage: ./generate_7x5_board.sh [options]"
#   echo "Options:"
#   echo "  -s, --square-length VAL  Square length in meters (default: $SQUARE_LENGTH)"
#   echo "  -m, --marker-length VAL  Marker length in meters (default: $MARKER_LENGTH)"
#   echo "  -d, --dictionary VAL     Dictionary name (default: $DICTIONARY)"
#   echo "  -p, --dpi VAL            DPI for the PNG image (default: $DPI)"
#   echo "  -g, --margin VAL         Margin in pixels (default: $MARGIN)"
#   echo "  -h, --help               Show this help message"
# }

# # Parse options
# while [[ $# -gt 0 ]]; do
#   case "$1" in
#     -s|--square-length)
#       SQUARE_LENGTH="$2"
#       shift 2
#       ;;
#     -m|--marker-length)
#       MARKER_LENGTH="$2"
#       shift 2
#       ;;
#     -d|--dictionary)
#       DICTIONARY="$2"
#       shift 2
#       ;;
#     -p|--dpi)
#       DPI="$2"
#       shift 2
#       ;;
#     -g|--margin)
#       MARGIN="$2"
#       shift 2
#       ;;
#     -h|--help)
#       show_help
#       exit 0
#       ;;
#     *)
#       echo "Unknown option: $1"
#       show_help
#       exit 1
#       ;;
#   esac
# done

# VENV_DIR=".venv"

# # Determine python executable path (direct execution avoids shell activation issues)
# if [ -f "$VENV_DIR/Scripts/python.exe" ]; then
#     PYTHON_EXE="$VENV_DIR/Scripts/python.exe"
# elif [ -f "$VENV_DIR/Scripts/python" ]; then
#     PYTHON_EXE="$VENV_DIR/Scripts/python"
# elif [ -f "$VENV_DIR/bin/python" ]; then
#     PYTHON_EXE="$VENV_DIR/bin/python"
# elif command -v python3 &>/dev/null; then
#     PYTHON_EXE="python3"
# else
#     PYTHON_EXE="python"
# fi

# echo "Generating 7x5 ChArUco board with parameters:"
# echo "  Squares: 7x5"
# echo "  Square Length: $SQUARE_LENGTH m"
# echo "  Marker Length: $MARKER_LENGTH m"
# echo "  Dictionary: $DICTIONARY"
# echo "  DPI: $DPI"
# echo "  Margin: $MARGIN px"

# "$PYTHON_EXE" generate_calib_pic.py "$SQUARES_X" "$SQUARES_Y" "$SQUARE_LENGTH" "$MARKER_LENGTH" "$DICTIONARY" --png --dpi "$DPI" --margin "$MARGIN"

python generate_calib_pic.py 7 5 0.0035 0.002 "$DICTIONARY" --png --dpi "$DPI" --margin "$MARGIN"
python generate_calib_pic.py 8 6 0.003 0.002 "$DICTIONARY" --png --dpi "$DPI" --margin "$MARGIN"
python generate_calib_pic.py 9 7 0.0025 0.0018 "$DICTIONARY" --png --dpi "$DPI" --margin "$MARGIN"
python generate_calib_pic.py 12 9 0.002 0.0015 "$DICTIONARY" --png --dpi "$DPI" --margin "$MARGIN"
