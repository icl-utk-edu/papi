cd c:\papi\src\win2k\winpmc
net stop winpmc
build -ceZ
copy /-Y /B sys\objchk\i386\winpmc.sys c:\winnt\system32\drivers
net start winpmc