# Microsoft Developer Studio Project File - Name="WinPAPI" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Static Library" 0x0104

CFG=WinPAPI - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "WinPAPI.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "WinPAPI.mak" CFG="WinPAPI - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "WinPAPI - Win32 Release" (based on "Win32 (x86) Static Library")
!MESSAGE "WinPAPI - Win32 Debug" (based on "Win32 (x86) Static Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
RSC=rc.exe

!IF  "$(CFG)" == "WinPAPI - Win32 Release"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 0
# PROP BASE Output_Dir "Release"
# PROP BASE Intermediate_Dir "Release"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 0
# PROP Output_Dir "Release"
# PROP Intermediate_Dir "Release"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /YX /FD /c
# ADD CPP /nologo /MT /W3 /GX /O2 /I "c:\papi\src\winpmc" /I "c:\papi\src\winpmc\sys" /I "." /I "..\winpmc" /I "..\winpmc\sys" /I "..\.." /D "WIN32" /D "NDEBUG" /D "_MBCS" /D "_LIB" /D "LANGUAGE_US" /FD /c
# SUBTRACT CPP /YX /Yc /Yu
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo

!ELSEIF  "$(CFG)" == "WinPAPI - Win32 Debug"

# PROP BASE Use_MFC 0
# PROP BASE Use_Debug_Libraries 1
# PROP BASE Output_Dir "Debug"
# PROP BASE Intermediate_Dir "Debug"
# PROP BASE Target_Dir ""
# PROP Use_MFC 0
# PROP Use_Debug_Libraries 1
# PROP Output_Dir "Debug"
# PROP Intermediate_Dir "Debug"
# PROP Target_Dir ""
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /YX /FD /GZ /c
# ADD CPP /nologo /MTd /W3 /Gm /GX /ZI /Od /I "." /I "..\winpmc" /I "..\winpmc\sys" /I "..\.." /D "WIN32" /D "_DEBUG" /D "_MBCS" /D "_LIB" /D "LANGUAGE_US" /FD /GZ /c
# SUBTRACT CPP /YX
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LIB32=link.exe -lib
# ADD BASE LIB32 /nologo
# ADD LIB32 /nologo

!ENDIF 

# Begin Target

# Name "WinPAPI - Win32 Release"
# Name "WinPAPI - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=cpuinfo.c
# End Source File
# Begin Source File

SOURCE=..\..\extras.c
# End Source File
# Begin Source File

SOURCE=..\..\multiplex.c
# End Source File
# Begin Source File

SOURCE=..\..\papi.c
# End Source File
# Begin Source File

SOURCE=..\..\papi_fwrappers.c
# End Source File
# Begin Source File

SOURCE=..\..\papi_hl.c
# End Source File
# Begin Source File

SOURCE=..\winpmc\pmclib.c
# End Source File
# Begin Source File

SOURCE=win32.c
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=cpuinfo.h
# End Source File
# Begin Source File

SOURCE=..\..\papi.h
# End Source File
# Begin Source File

SOURCE=..\..\papi_internal.h
# End Source File
# Begin Source File

SOURCE=..\..\papiStdEventDefs.h
# End Source File
# Begin Source File

SOURCE=..\..\papiStdEventDescrs.h
# End Source File
# Begin Source File

SOURCE=..\..\papiStdEventNames.h
# End Source File
# Begin Source File

SOURCE=..\..\papiStrings.h
# End Source File
# Begin Source File

SOURCE=..\..\papiStrings_US.h
# End Source File
# Begin Source File

SOURCE=..\winpmc\pmclib.h
# End Source File
# Begin Source File

SOURCE=win32.h
# End Source File
# Begin Source File

SOURCE=win_extras.h
# End Source File
# End Group
# Begin Source File

SOURCE="..\..\..\..\Program Files\Microsoft Visual Studio\Vc98\Lib\Winmm.lib"
# End Source File
# End Target
# End Project
