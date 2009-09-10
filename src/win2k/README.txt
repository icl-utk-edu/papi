/* 
* File:    README.TXT
* Author:  dan terpstra
*          terpstra@cs.utk.edu
*/  

WinPAPI Version 2.1 Release Notes
Feb, 2002

NOTES:
The stuff in this folder is a LIMITED FUNCTION RELEASE of PAPI for Windows 2000.
It runs on Windows NT, 2000 and XP.
It requires Administrator priveleges to install the kernel driver.
It works on both UniProcessor and MultiProcessor builds of Windows.
It binds itself to a single processor on MP systems.
It measures system-wide events, not thread or process specific.
It currently DOES NOT support multiplexing, overflow, or profiling.

With those caveats out of the way, here's what we've got:

DIRECTORIES INCLUDED:
- ftests:	Compaq Visual Fortran projects for PAPI test examples. 
- help:	html files that document what's in the install.
- MatLab:	projects and files to build a MATLAB mex file to exercise PAPI FLOPS.
- shell:	contains various pieces for building a shell application for exercising the WinPAPI Library.
- substrate: the stuff for building a static WinPAPI library to link to your application
- tests:	C projects for PAPI test examples.
- winpmc:	stuff to build and install the Windows PMC system service (kernel driver) 
	that provides access to the performance counters.

TOOLS REQUIRED:
- The tests, shell and substrate require Microsoft Visual C++ version 6 or better.
- The Fortran tests require Compaq Visual Fortran V 6.6 or greater.
- Workspaces and projects for all of these codes are included in the respective directories.
- The WinPMC driver requires the Microsoft NTDDK development environment, 
	which depends on Visual C++, and is available as a free download 
	from the Microsoft website.
