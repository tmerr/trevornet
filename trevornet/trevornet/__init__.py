def test(verbose=False):
    import os, nose

    from . import tests
    directory = os.path.dirname(tests.__file__)

    argv = ['nosetests', '--exe', directory]

    try:
        return nose.main(argv=argv)
    except SystemExit as e:
        return e.code
