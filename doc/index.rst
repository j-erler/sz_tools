sz_tools
=========

**sz_tools** is a collection of codes that are useful for studies of SZ-effect data e.g. from 
Planck, as well as general studies of galaxy clusters. Among its features are SED computation, 
pressure profile computation and projection, ILC algorithems, statistical tools and wrappers 
commonly used astropy functions. sz_tools is fully implemented in python 3.

This documentation provides an overview on the installation process and the available 
functions that are part of sz_tools. Practical examples are provided in jupyter 
notebooks that can be found in the /examples directory.

sz_tools is being actively developed on `GitHub <https://github.com/j-erler/sz_tools>`_.

.. image:: https://img.shields.io/badge/GitHub-j--erler%2Fsz__tools-blue.svg?style=flat
    :target: https://github.com/j-erler/sz_tools
.. image:: https://img.shields.io/badge/docs-passing-green.svg?style=flat
    :target: https://sz-tools.readthedocs.io/en/latest/index.html#
.. image:: https://img.shields.io/badge/license-MIT-red.svg?style=flat
    :target: https://github.com/j-erler/sz_tools/blob/master/LICENSE
    
.. toctree::
   :maxdepth: 2
   :caption: Contents:


Acknowledgement
---------------

Please cite `Erler, Basu, Chluba & Bertoldi (2018)
<https://arxiv.org/abs/1809.06446>`_ if you find the ILC implementation that is 
part of sz_tools useful in your research.
The BibTeX entry for the paper is::

	@ARTICLE{2018MNRAS.476.3360E,
		author = {{Erler}, J. and {Basu}, K. and {Chluba}, J. and {Bertoldi}, F.},
		 title = "{Planck's view on the spectrum of the Sunyaev-Zeldovich effect}",
	       journal = {\mnras},
		  year = 2018,
		volume = 476,
		 pages = {3360-3381},
		   doi = {10.1093/mnras/sty327},
	}

Installation
---------

.. toctree::
   :maxdepth: 2

   install


Reference
---------

.. toctree::
   :maxdepth: 2

   code

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

License
-------

.. toctree::
   :maxdepth: 1

   license
