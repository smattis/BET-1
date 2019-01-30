BET
===
[![Build Status](https://travis-ci.org/UT-CHG/BET.svg?branch=master)](https://travis-ci.org/UT-CHG/BET) [![DOI](https://zenodo.org/badge/18813599.svg)](https://zenodo.org/badge/latestdoi/18813599)


BET is in active development. Hence, some features are still being added and you may find bugs we have overlooked. If you find something please report these problems to us through GitHub so that we can fix them. Thanks! 

Please note that we are using continuous integration and issues for bug tracking.

## Butler, Estep, Tavener method

This code has been documented with sphinx. the documentation is available online ta http://ut-chg.github.io/bet. to build documentation run 
``make html`` in the ``doc/`` folder.
to build/update the documentation use the following commands::

    sphinx-apidoc -f -o doc bet
    cd doc/
    make html
    make html

this creates the relevant documentation at ``bet/gh-pages/html``. to change the build location of the documentation you will need to update ``doc/makefile``.

you will need to run sphinx-apidoc and reinstall bet anytime a new module or method in the source code has been added. if only the `*.rst` files have changed then you can simply run ``make html`` twice in the doc folder.

useful scripts are contained in ``examples/``

tests
-----

to run tests in serial call::

    nosetests

to run tests in parallel call::

    mpirun -np nproc nosetests

(make sure to have run `apt-get install mpich libmpich-dev`)

dependencies
------------

`bet` requires the following packages:

1. [numpy](http://www.numpy.org/)
2. [scipy](http://www.scipy.org/)
3. [nose](https://nose.readthedocs.org/en/latest/)
4. [pydoe](https://pythonhosted.org/pydoe/)
5. [matplotlib](http://matplotlib.org/)

(note: you may need to set `~/.config/matplotlib/matplotlibrc` to include `backend:agg` if there is no `display` port in your environment). 
