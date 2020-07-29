from patchCreater import PatchCreater
import argparse
import re

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", help="$HOME/Desktop/data/Abdomen/case_00/imaging.nii.gz")
    parser.add_argument("modelweight_path", help="$HOME/Desktop/data/modelweight/Abdomen/32-32-32/mask/latest.pkl")
    parser.add_argument("save_path", help="$HOME/Desktop/data/patch/512-512-32/")
    parser.add_argument("--output_layer", default=1, type=int)
    parser.add_argument("--input_size", help="32-32-32", default="32-32-32")
    parser.add_argument("--plane_size", help="512-512", default="512-512")
    parser.add_argument("--overlap", default=1, type=int, help="Overlap in axial direction. Default 1")
    parser.add_argument("--gpu_ids", help="0 1", nargs="*", default=0, type=int)

    args = parser.parse_args()

    return args

def main(args):
    matchobj = re.match("([0-9]+)-([0-9]+)-([0-9]+)", args.input_size)
    if matchobj is None:
        print("[ERROR] Invalid patch size : {}".format(args.input_size))
        sys.exit()

    input_size = [int(s) for s in matchobj.groups()]

    matchobj = re.match("([0-9]+)-([0-9]+)", args.plane_size)
    if matchobj is None:
        print("[ERROR] Invalid patch size : {}".format(args.plane_size))
        sys.exit()

    plane_size = [int(s) for s in matchobj.groups()]

    pc = PatchCreater(
            image_path = args.image_path, 
            modelweight_path = args.modelweight_path, 
            output_layer = args.output_layer,
            input_size = input_size,
            plane_size = plane_size,
            overlap = args.overlap,
            gpu_ids = args.gpu_ids
            )

    pc.execute()
    pc.save(args.save_path)

if __name__ == "__main__":
    args = parseArgs()
    main(args)
