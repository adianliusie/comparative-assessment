import argparse
import sys
import os

from src.utils.post_processing import save_combined_json, delete_leftover_files

if __name__ == '__main__':
    # build parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='where to load and save combined outputs')

    args = parser.parse_args()
    save_combined_json(args.path)  

    num_files = len(os.listdir(args.path))
    if num_files <= 1:
        print('No files to combine/delete!')
    else:
        print(f"there are {num_files} files")
        out = input('do you want to delete all leftover individual files?  type Y to confirm: ')
        if out == 'Y':
            delete_leftover_files(args.path)
        else:
            print('files not deleted')

# import sys
# from src.utils.post_processing import save_combined_json

# if __name__ == '__main__':
#     path = sys.argv[1]
#     save_combined_json(path)

    
