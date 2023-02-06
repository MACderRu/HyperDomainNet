import click
import subprocess

from pathlib import Path


def download_curl(source: str, destination: str) -> None:
    # urllib has troubles with dropbox
    subprocess.run(
        ['curl', '-L', '-k', source, '-o', destination]
    )


def download_gdrive(file_id: str, destination: str) -> None:
    subprocess.run(['gdown', '--id', file_id, '-O', destination])
    
    
def unzip(path: str, res_path: str = None):
    command = ['unzip', path]
    
    if res_path is not None:
        command += ['-d', res_path]
    subprocess.run(command)


def bzip2(path: str):
    subprocess.run(['bzip2', '-d', path])

    
def rm_file(path: str):
    subprocess.run(['rm', path])
    


SOURCES = {
    'stylegan2': {
        'id': '1Yr7KuD959btpmcKGAUsbAk5rPjX2MytK',
        'name': 'stylegan2-ffhq-config-f.pt'
    },
    'dlib': {
        'link': 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
        'name': 'shape_predictor_68_face_landmarks.dat.bz2'
    },
    'restyle_psp': {
        'id': '1nbxCIVw9H3YnQsoIPykNEFwWJnHVHlVd',
        'name': 'restyle_psp_ffhq_encode.pt'
    },
    'e4e': {
        'id': '1o6ijA3PkcewZvwJJ73dJ0fxhndn0nnh7',
        'name': 'e4e_ffhq_encode.pt'
    },
    'checkpoints': {
        'link': 'https://www.dropbox.com/s/r8816i09t9n94hy/checkpoints.zip?dl=0',
        'name': 'checkpoints.zip'
    }
}


class Setup:    
    def __init__(self):
        self.destination = Path(__file__).parent / 'pretrained'
        self.destination.mkdir(exist_ok=True)
        
    def _download(self, data):
        
        file_dest = str(self.destination / data['name'])
        
        if 'link' in data:
            download_curl(data['link'], file_dest)
        elif 'id' in data:
            download_gdrive(data['id'], file_dest)
        
        if file_dest.endswith('bz2'):
            bzip2(file_dest)
            rm_file(file_dest)
        
        if file_dest.endswith('zip'):
            unzip(file_dest, str(self.destination))
            rm_file(file_dest)
        
    def setup(self, values):
        for value in values:
            self._download(SOURCES[value])
            
            
@click.command()
@click.argument('value', default=None, nargs=-1)
def main(value):
    setuper = Setup()
    values = value if value else SOURCES.keys()
    setuper.setup(values)
    

if __name__ == '__main__':
    main()