import click


@click.group()
def cli():

    print("WELCOME.")
    return 'Running'
