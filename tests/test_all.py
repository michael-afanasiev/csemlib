from click.testing import CliRunner

from csemlib import csemlib


def test_main():
    '''Dumb test.'''

    runner = CliRunner()
    result = runner.invoke(csemlib.cli)
    assert True
