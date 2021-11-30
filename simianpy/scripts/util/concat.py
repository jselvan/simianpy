from pathlib import Path
import shutil
import click

@click.command('concat')
@click.argument('files', nargs=-1)
@click.option('-o','--output', default='concat.bin')
@click.option('-f','--force',is_flag=True,default=False)
def concat(files, output, force):
    output = Path(output)
    files = [Path(file) for file in files]
    if any(not file.is_file() for file in files):
        raise FileNotFoundError(file)
    if output.is_file and not force:
        raise FileExistsError('Use -f to overwrite existing file')

    print('Concatenating')
    for idx, file in enumerate(files):
        print('\t', idx+1, file.name)
    print('Into','\n\t',output.name)

    with open(output, 'wb') as outputfile:
        for file in files:
            with open(file, 'rb') as inputfile:
                shutil.copyfileobj(inputfile, outputfile)
    
    print('Concatenation complete')