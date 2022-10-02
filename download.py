import tarfile
import argparse
import subprocess
from pathlib import Path


SOURCES = {
    'StyleGAN2': 'https://www.dropbox.com/s/ovt5y7yfn2odwbf/StyleGAN2_weights.zip?dl=0',
    'DLIB': 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
}


def download(source: str, destination: str) -> None:
    # urllib has troubles with dropbox
    subprocess.run(
        ['curl', '-L', '-k', source, '-o', destination]
    )


def unzip(path: str, res_path: str):
    subprocess.run(
        ['unzip', path, '-d', res_path]
    )


def bzip2(path: str):
    subprocess.run(
        ['bzip2', '-d', path]
    )

def rm_file(path: str):
    subprocess.run(
        ['rm', path]
    )


def load_stylegan2_weights(source, destdir, tmp_name='test.zip'):
    if (destdir / 'StyleGAN2').exists():
        print('[StyleGAN2] weights are already downloaded')
        return
    
    print(f'[StyleGAN2] Downloading...')
    download(source, str(destdir / tmp_name))
    unzip(str(destdir / tmp_name), str(destdir))
    rm_file(str(destdir / tmp_name))


def load_dlib_model(source, destdir):
    file_name = source.split('/')[-1]
    if (destdir / file_name).exists():
        print('[DLIB] model is already downloaded')
        return
    
    print(f'[DLIB] Downloading...')
    download(source, str(destdir / file_name))
    bzip2(str(destdir / file_name))

    
def load_clip_models():
    import clip
    for model_name in ['ViT-B/32', 'ViT-B/16', 'ViT-L/14']:
        clip.load(model_name)
        print(f'[CLIP] {model_name} is loaded')
        
    
def main():
    destination = Path(__file__).parent / 'pretrained'
    destination.mkdir(exist_ok=True)    
    load_stylegan2_weights(SOURCES['StyleGAN2'], destination)
    load_dlib_model(SOURCES['DLIB'], destination)
    load_clip_models()
    


if __name__ == '__main__':
    main()

