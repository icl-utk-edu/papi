/* 
* File:    README.TXT
* Author:  dan terpstra
*          terpstra@cs.utk.edu
*/  

WinPAPI Version 0.2 Release Notes
June, 2001

NOTE:
The stuff in this folder is a PRELIMINARY RELEASE of PAPI for Windows 2000.
It is STILL UNDER DEVELOPMENT.
It is NOT INTENDED to be installed on user systems.
It REQUIRES changes to the base PAPI sources that have not yet been committed to CVS.

With those caveats out of the way, here's what we've got:

DIRECTORIES INCLUDED:
- shell:	contains various pieces for building a test shell application 
	for exercising the WinPAPI Library.
- substrate: the stuff for building a static WinPAPI library to link to your application
- winpmc:	stuff to build and install the Windows PMC system service (kernel driver) 
	that provides access to the performance counters.

TOOLS REQUIRED:
- The shell and substrate require Microsoft Visual C++ version 6 or better.
- Workspaces and projects for both of these codes are included in the respective directories.
- The WinPMC driver requires the Microsoft NTDDK development environment, 
	which depends on Visual C++, and is available as a free download 
	from the Microsoft website.
