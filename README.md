PAPI: Performance Application Programming Interface
===================================================

** Innovative Computing Lab **

** University of Tennessee, Knoxville, TN **

***
[TOC]
***

About
-----

PAPI provides the tool designer and application engineer with a
consistent interface and methodology for use of the performance
counter hardware found in most major microprocessors. PAPI enables
software engineers to see, in near real time, the relation between
software performance and processor events.

In addition, PAPI provides access to a collection of components that
expose performance measurement opportunites across the hardware and
software stack.


Getting Started
---------------

If this is the first file you've opened in the PAPI tree, we'll try to give you
a few tips on where to go from here.

* Read the license found in LICENSE.txt. It's pretty short, and not very 
  restrictive, but it'll give you an idea of what you can and can't do with the
  PAPI sources.
* Visit the website at: <http://icl.utk.edu/papi>
  There you can find late-breaking news that may be more current than in these
  files. You can also find documentation in a greater variety of formats than
  in the papi/doc/ directory.
* Sign up for the PAPI mailing list(s). Instructions are on our home page.
* Read the RELEASENOTES.txt file to get an idea of what's new in the current release.


Downloading PAPI
----------------

* Clone the PAPI repository the first time with the following command:  
   `> git clone https://bitbucket.org/icl/papi.git`

* This creates a complete copy of the papi git repository on your computer 
  in a folder called 'papi'.

* To make sure your copy is up to date with the repository:  
   `> cd papi`  
   `> git pull  https://bitbucket.org/icl/papi.git`


Installing PAPI
---------------

To install PAPI on your system:

* Find the section in INSTALL.txt that pertains to your hardware and operating
  system.
* Follow the directions to install required components and build the PAPI
  libraries.
* Run the test suite when you are finished to verify that everything went ok.
  NOTE: Although we make every attempt to get all tests to PASS or SKIP on all
  platforms, there are occasional instances of FAILures due to excessively 
  tight compliance thresholds or platform idiosyncrasies. Don't panic if one
  or two tests FAIL. Contact us with complete output and we'll see what we can do.


Using PAPI
----------

To use PAPI in your own programs:

* Read the PAPI Overview found at:
	http://icl.utk.edu/projects/papi/wiki/Main_Page.
* Try out the utility programs in /utils to see what's in your system.
* Try a test program. Source for a number of tests in both C and FORTRAN is
  available in the src/tests/ and src/ftests/ directories. Find a program
  that's similar to what you want to do. Make sure you can build it and run it.
* Write a test program of your own, exercising the PAPI events and features of
  interest to you.
* Go for broke. Fold PAPI calls into your sources and see what you can learn.


Bugs and Questions
------------------

* Visit our FAQ at: <http://icl.utk.edu/papi/faq/>
  or read a snapshot of the FAQ in papi/PAPI_FAQ.html
* Subscribe to the PAPI mailing list at:
  <https://groups.google.com/a/icl.utk.edu/forum/#!forum/ptools-perfapi>
* Read historical postings to the list.
* Post questions to the list.