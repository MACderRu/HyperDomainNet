import click
import tarfile
import argparse
import subprocess

from pathlib import Path


SOURCES = {
    'StyleGAN2': 'https://www.dropbox.com/s/ovt5y7yfn2odwbf/StyleGAN2_weights.zip?dl=0',
    'DLIB': 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
    'checkpoints': 'single': 'https://www.dropbox.com/s/nigsjd0di41p950/checkpoints.zip?dl=0',
    'restyle': 'https://www.dropbox.com/s/b3atzfkx0upbx10/restyle_psp_ffhq_encode.pt?dl=0'
}


def download(source: str, destination: str) -> None:
    # urllib has troubles with dropbox
    subprocess.run(
        ['curl', '-L', '-k', source, '-o', destination]
    )


def unzip(path: str, res_path: str = None):
    command = ['unzip', path]
    
    if res_path is not None:
        command += ['-d', res_path]
    subprocess.run(command)


def bzip2(path: str):
    subprocess.run(['bzip2', '-d', path])

    
def rm_file(path: str):
    subprocess.run(['rm', path])


def load_stylegan2_weights(tmp_name='test.zip'):
    source = SOURCES['StyleGAN2']
    destdir = Path(__file__).parent / 'pretrained'
    if (destdir / 'StyleGAN2').exists():
        print('[StyleGAN2] weights are already downloaded')
        return
    
    print(f'[StyleGAN2] Downloading...')
    download(source, str(destdir / tmp_name))
    unzip(str(destdir / tmp_name), str(destdir))
    rm_file(str(destdir / tmp_name))


def load_dlib_model():
    source = SOURCES['DLIB']
    destination = Path(f"pretrained/{source.split('/')[-1]}")
    
    if Path('pretrained/shape_predictor_68_face_landmarks.dat').exists():
        print('[DLIB] model is already downloaded')
        return
    
    print(f'[DLIB] Downloading...')
    download(source, str(destination))
    bzip2(str(destination))

    
def load_clip_models():
    import clip
    for model_name in ['ViT-B/32', 'ViT-B/16', 'ViT-L/14']:
        clip.load(model_name)
        print(f'[CLIP] {model_name} is loaded')

        
def load_restyle_weights():
    path = f'pretrained/restyle_psp_ffhq_encode.pt'
    if Path(path).exists():
        print('[ReStyle] weights are already downloaded')
        return
    
    print(f'[ReStyle] Downloading...')
    download(SOURCES['restyle'], f'pretrained/restyle_psp_ffhq_encode.pt')

        
def download_checkpoints(keys=None):
    download(SOURCES['checkpoints'], 'checkpoints.zip')
    unzip('checkpoints.zip')
    rm_file('checkpoints.zip')


loaders = {
    'stylegan2': load_stylegan2_weights,
    'dlib': load_dlib_model,
    'clip': load_clip_models,
    'checkpoints': download_checkpoints,
    'restyle': load_restyle_weights
}


@click.command()
@click.option('--load_type', default=None)
def main(load_type):
    destination = Path(__file__).parent / 'pretrained'
    destination.mkdir(exist_ok=True)
    
    if load_type is None:
        for name, fn in loaders.items():
            fn()
    else:
        loaders[load_type]()
    

if __name__ == '__main__':
    main()
