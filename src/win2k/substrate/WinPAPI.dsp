# Microsoft Developer Studio Project File - Name="WinPAPI" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Dynamic-Link Library" 0x0102

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
!MESSAGE "WinPAPI - Win32 Release" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE "WinPAPI - Win32 Debug" (based on "Win32 (x86) Dynamic-Link Library")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
F90=df.exe
MTL=midl.exe
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
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MT /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "WinPAPI_EXPORTS" /YX /FD /c
# ADD CPP /nologo /MT /W3 /GX /O2 /I "." /I "..\.." /I "..\winpmc" /I "..\winpmc\sys" /D "NDEBUG" /D "WIN32" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "FORTRANCAPS" /D "WinPAPI_EXPORTS" /D "NO_VARARG_MACRO" /D "NO_FUNCTION_MACRO" /YX /FD /D SUBSTRATE=<perfctr-p3.h> /c
# ADD BASE MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /machine:I386
# ADD LINK32 winmm.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib libcmt.lib psapi.lib /nologo /dll /machine:I386
# Begin Special Build Tool
SOURCE="$(InputPath)"
PostBuild_Desc=Copying DLL to target directories
PostBuild_Cmds=copy release\WinPAPI.dll ..\shell\release	copy release\WinPAPI.dll c:\winnt\system32
# End Special Build Tool

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
# PROP Ignore_Export_Lib 0
# PROP Target_Dir ""
# ADD BASE CPP /nologo /MTd /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "WinPAPI_EXPORTS" /YX /FD /GZ /c
# ADD CPP /nologo /MTd /W3 /Gm /GX /ZI /Od /I "." /I "..\.." /I "..\winpmc" /I "..\winpmc\sys" /D "_DEBUG" /D "WIN32" /D "_WINDOWS" /D "_MBCS" /D "_USRDLL" /D "FORTRANCAPS" /D "WinPAPI_EXPORTS" /D "NO_VARARG_MACRO" /D "NO_FUNCTION_MACRO" /YX /FD /GZ /D SUBSTRATE=<perfctr-p3.h> /c
# ADD BASE MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /dll /debug /machine:I386 /pdbtype:sept
# ADD LINK32 winmm.lib kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib libcmt.lib psapi.lib /nologo /dll /debug /machine:I386 /nodefaultlib:"libcmtd.lib" /pdbtype:sept
# Begin Special Build Tool
SOURCE="$(InputPath)"
PostBuild_Desc=Installing debug dll
PostBuild_Cmds=copy debug\WinPAPI.dll c:\winnt\system32
# End Special Build Tool

!ENDIF 

# Begin Target

# Name "WinPAPI - Win32 Release"
# Name "WinPAPI - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=.\cpuinfo.c
# End Source File
# Begin Source File

SOURCE=..\..\extras.c
# End Source File
# Begin Source File

SOURCE="..\..\linux-memory.c"
# End Source File
# Begin Source File

SOURCE=..\..\linux.c
# End Source File
# Begin Source File

SOURCE=..\..\multiplex.c
# End Source File
# Begin Source File

SOURCE=..\..\p3_events.c
# End Source File
# Begin Source File

SOURCE=..\..\papi.c
# End Source File
# Begin Source File

SOURCE=..\..\papi_data.c
# End Source File
# Begin Source File

SOURCE=..\..\papi_fwrappers.c
# End Source File
# Begin Source File

SOURCE=..\..\papi_hl.c
# End Source File
# Begin Source File

SOURCE=..\..\papi_internal.c
# End Source File
# Begin Source File

SOURCE=..\..\papi_preset.c
# End Source File
# Begin Source File

SOURCE="..\..\perfctr-p3.c"
# End Source File
# Begin Source File

SOURCE=..\winpmc\pmclib.c
# End Source File
# Begin Source File

SOURCE=..\..\threads.c
# End Source File
# Begin Source File

SOURCE=.\WinPAPI.def
# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\cpuinfo.h
# End Source File
# Begin Source File

SOURCE=..\..\papi.h
# End Source File
# Begin Source File

SOURCE=..\..\papi_internal.h
# End Source File
# Begin Source File

SOURCE=..\..\papi_protos.h
# End Source File
# Begin Source File

SOURCE=..\..\papiStdEventDefs.h
# End Source File
# Begin Source File

SOURCE=..\..\papiStrings.h
# End Source File
# Begin Source File

SOURCE=..\winpmc\pmclib.h
# End Source File
# Begin Source File

SOURCE=.\win_extras.h
# End Source File
# End Group
# End Target
# End Project
