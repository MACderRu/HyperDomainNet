import os
import subprocess

from pathlib import Path


def get_download_model_command(save_path, file_id, file_name):
    """ Get wget download command for downloading the desired model and save to directory ../pretrained_models. """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    url = r"""wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id={FILE_ID}' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id={FILE_ID}" -O {SAVE_PATH}/{FILE_NAME} && rm -rf /tmp/cookies.txt""".format(FILE_ID=file_id, FILE_NAME=file_name, SAVE_PATH=save_path)
    return url  


MODEL_PATHS = {
    "ffhq_encode": {"id": "1sw6I2lRIB0MpuJkpc8F5BJiSZrc0hjfE", "name": "restyle_psp_ffhq_encode.pt"},
    "cars_encode": {"id": "1zJHqHRQ8NOnVohVVCGbeYMMr6PDhRpPR", "name": "restyle_psp_cars_encode.pt"},
    "church_encode": {"id": "1bcxx7mw-1z7dzbJI_z7oGpWG1oQAvMaD", "name": "restyle_psp_church_encode.pt"},
    "horse_encode": {"id": "19_sUpTYtJmhSAolKLm3VgI-ptYqd-hgY", "name": "restyle_e4e_horse_encode.pt"},
    "afhq_wild_encode": {"id": "1GyFXVTNDUw3IIGHmGS71ChhJ1Rmslhk7", "name": "restyle_psp_afhq_wild_encode.pt"},
    "toonify": {"id": "1GtudVDig59d4HJ_8bGEniz5huaTSGO_0", "name": "restyle_psp_toonify.pt"}
}


if __name__ == "__main__":
    exp = 'ffhq_encode'
    path = MODEL_PATHS[exp]
    path_to_save = Path(os.getcwd()).resolve() / 'pretrained'
    download_command = get_download_model_command(str(path_to_save), file_id=path["id"], file_name=path["name"]) 
    
    if not os.path.exists(path_to_save / path['name']) or os.path.getsize(path_to_save / path['name']) < 1000000:
        print(f'Downloading ReStyle model for {exp}...')
        subprocess.run(f"wget {download_command}", shell=True, check=True)
        
        # if google drive receives too many requests, we'll reach the quota limit and be unable to download the model
        if os.path.getsize(EXPERIMENT_ARGS['model_path']) < 1000000:
            raise ValueError("Pretrained model was unable to be downloaded correctly!")
        else:
            print('Done.')
    else:
        print(f'ReStyle model for {exp} already exists!')