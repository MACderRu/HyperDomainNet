import tarfile
import argparse
import subprocess
from pathlib import Path


SOURCES = {
    'mnist': 'https://www.dropbox.com/s/rzurpt5gzb14a1q/pretrained_mnist.tar',
    'anime': 'https://www.dropbox.com/s/9aveavgbluvjeu6/pretrained_anime.tar',
    'biggan': 'https://www.dropbox.com/s/zte4oein08ajsij/pretrained_biggan.tar',
    'proggan': 'https://www.dropbox.com/s/707xjn1rla8nwqc/pretrained_proggan.tar',
    'stylegan2': 'https://www.dropbox.com/s/c3aaq7i6soxmpzu/pretrained_stylegan2_ffhq.tar',
}


def download(source: str, destination: Path) -> None:
    tmp_tar = str(destination / '.tmp.tar')
    # urllib has troubles with dropbox
    subprocess.run(
        ['curl', '-L', '-k', source, '-o', tmp_tar]
    )
    tar_file = tarfile.open(tmp_tar, mode='r')
    tar_file.extractall(destination)
    subprocess.run(
        ['rm', tmp_tar]
    )


def main():
    parser = argparse.ArgumentParser(description='Pretrained models loader')
    parser.add_argument('--models', nargs='+', type=str,
                        choices=list(SOURCES.keys()) + ['all'], default=['all'])
    parser.add_argument('--out', type=str, help='root out dir')
    args = parser.parse_args()

    if args.out is None:
        args.out = Path(__file__).parent / 'models'
    else:
        args.out = Path(args.out)

    if not Path(args.out).exists():
        Path(args.out).mkdir()

    models = args.models
    if 'all' in models:
        models = list(SOURCES.keys())

    for model in set(models):
        source = SOURCES[model]
        print(f'downloading {model}\nfrom {source}')
        download(source, args.out)


if __name__ == '__main__':
    main()
