# Microsoft Developer Studio Project File - Name="WinPAPIShell" - Package Owner=<4>
# Microsoft Developer Studio Generated Build File, Format Version 6.00
# ** DO NOT EDIT **

# TARGTYPE "Win32 (x86) Application" 0x0101

CFG=WinPAPIShell - Win32 Debug
!MESSAGE This is not a valid makefile. To build this project using NMAKE,
!MESSAGE use the Export Makefile command and run
!MESSAGE 
!MESSAGE NMAKE /f "WinPAPIShell.mak".
!MESSAGE 
!MESSAGE You can specify a configuration when running NMAKE
!MESSAGE by defining the macro CFG on the command line. For example:
!MESSAGE 
!MESSAGE NMAKE /f "WinPAPIShell.mak" CFG="WinPAPIShell - Win32 Debug"
!MESSAGE 
!MESSAGE Possible choices for configuration are:
!MESSAGE 
!MESSAGE "WinPAPIShell - Win32 Release" (based on "Win32 (x86) Application")
!MESSAGE "WinPAPIShell - Win32 Debug" (based on "Win32 (x86) Application")
!MESSAGE 

# Begin Project
# PROP AllowPerConfigDependencies 0
# PROP Scc_ProjName ""
# PROP Scc_LocalPath ""
CPP=cl.exe
MTL=midl.exe
RSC=rc.exe

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

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
# ADD BASE CPP /nologo /W3 /GX /O2 /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /Yu"stdafx.h" /FD /c
# ADD CPP /nologo /MT /W3 /GX /O2 /I "..\..\tests" /I "..\..\\" /I "..\substrate" /D "WIN32" /D "NDEBUG" /D "_WINDOWS" /D "_MBCS" /FR /Yu"stdafx.h" /FD /c
# ADD BASE MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "NDEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "NDEBUG"
# ADD RSC /l 0x409 /d "NDEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:windows /machine:I386
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:windows /machine:I386 /nodefaultlib:"libcmtd.lib"

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

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
# ADD BASE CPP /nologo /W3 /Gm /GX /ZI /Od /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /Yu"stdafx.h" /FD /GZ /c
# ADD CPP /nologo /MTd /W3 /Gm /GX /ZI /Od /I "..\..\\" /I "..\substrate" /D "WIN32" /D "_DEBUG" /D "_WINDOWS" /D "_MBCS" /FR /Yu"stdafx.h" /FD /GZ /c
# ADD BASE MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD MTL /nologo /D "_DEBUG" /mktyplib203 /win32
# ADD BASE RSC /l 0x409 /d "_DEBUG"
# ADD RSC /l 0x409 /d "_DEBUG"
BSC32=bscmake.exe
# ADD BASE BSC32 /nologo
# ADD BSC32 /nologo
LINK32=link.exe
# ADD BASE LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib /nologo /subsystem:windows /debug /machine:I386 /pdbtype:sept
# ADD LINK32 kernel32.lib user32.lib gdi32.lib winspool.lib comdlg32.lib advapi32.lib shell32.lib ole32.lib oleaut32.lib uuid.lib odbc32.lib odbccp32.lib c:\papi\src\win2k\shell\libatlas.a /nologo /subsystem:windows /debug /machine:I386 /nodefaultlib:"libcmtd.lib libcmt.lib" /pdbtype:sept
# SUBTRACT LINK32 /nodefaultlib

!ENDIF 

# Begin Target

# Name "WinPAPIShell - Win32 Release"
# Name "WinPAPIShell - Win32 Debug"
# Begin Group "Source Files"

# PROP Default_Filter "cpp;c;cxx;rc;def;r;odl;idl;hpj;bat"
# Begin Source File

SOURCE=..\..\tests\avail.c

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

# ADD CPP /I "c:\papi\src"
# SUBTRACT CPP /YX /Yc /Yu

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

# SUBTRACT CPP /YX /Yc /Yu

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\tests\clockres.c

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

# ADD CPP /I "c:\papi\src"
# SUBTRACT CPP /YX /Yc /Yu

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

# SUBTRACT CPP /YX /Yc /Yu

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\tests\cost.c

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

# ADD CPP /I "c:\papi\src"
# SUBTRACT CPP /YX /Yc /Yu

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

# SUBTRACT CPP /YX /Yc /Yu

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\dgemm_test.c
# SUBTRACT CPP /YX /Yc /Yu
# End Source File
# Begin Source File

SOURCE=..\..\tests\do_loops.c

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

# ADD CPP /I "c:\papi\src"
# SUBTRACT CPP /YX /Yc /Yu

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

# SUBTRACT CPP /YX /Yc /Yu

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\tests\fifth.c

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

# ADD CPP /I "c:\papi\src"
# SUBTRACT CPP /YX /Yc /Yu

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

# SUBTRACT CPP /YX /Yc /Yu

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\tests\first.c

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

# ADD CPP /I "c:\papi\src"
# SUBTRACT CPP /YX /Yc /Yu

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

# SUBTRACT CPP /YX /Yc /Yu

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\tests\flops.c

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

# ADD CPP /I "c:/papi/src" /D "PAPI_DEBUG"
# SUBTRACT CPP /YX /Yc /Yu

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

# ADD CPP /D "PAPI_DEBUG"
# SUBTRACT CPP /YX /Yc /Yu

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\tests\fourth.c

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

# ADD CPP /I "c:\papi\src"
# SUBTRACT CPP /YX /Yc /Yu

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

# SUBTRACT CPP /YX /Yc /Yu

!ENDIF 

# End Source File
# Begin Source File

SOURCE="..\..\tests\high-level.c"

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

# ADD CPP /I "c:\papi\src"
# SUBTRACT CPP /YX /Yc /Yu

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

# SUBTRACT CPP /YX /Yc /Yu

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\Nils_system_info.c

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

# SUBTRACT CPP /I "..\..\tests" /YX /Yc /Yu

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

# SUBTRACT CPP /YX /Yc /Yu

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\tests\nineth.c

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

# ADD CPP /I "c:\papi\src"
# SUBTRACT CPP /YX /Yc /Yu

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

# SUBTRACT CPP /YX /Yc /Yu

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\PAPI_Errors.c

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

# ADD CPP /I "c:/papi/src" /I "c:/papi/src/tests"
# SUBTRACT CPP /YX /Yc /Yu

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

# SUBTRACT CPP /YX /Yc /Yu

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\tests\second.c

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

# ADD CPP /I "c:\papi\src"
# SUBTRACT CPP /YX /Yc /Yu

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

# SUBTRACT CPP /YX /Yc /Yu

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\StdAfx.c
# ADD CPP /Yc"stdafx.h"
# End Source File
# Begin Source File

SOURCE=.\StringsAndLabels.c

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

# ADD CPP /I "c:/papi/src" /I "c:/papi/src/tests"
# SUBTRACT CPP /YX /Yc /Yu

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

# SUBTRACT CPP /YX /Yc /Yu

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\tests\tenth.c

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

# ADD CPP /I "c:\papi\src"
# SUBTRACT CPP /YX /Yc /Yu

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

# SUBTRACT CPP /YX /Yc /Yu

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\test_get_cycles.c

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

# ADD CPP /I "c:/papi/src"
# SUBTRACT CPP /YX /Yc /Yu

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

# SUBTRACT CPP /YX /Yc /Yu

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\tests\test_utils.c

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

# ADD CPP /I "c:/papi/src"
# SUBTRACT CPP /YX /Yc /Yu

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

# SUBTRACT CPP /YX /Yc /Yu

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\..\tests\third.c

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

# ADD CPP /I "c:/papi/src"
# SUBTRACT CPP /YX /Yc /Yu

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

# SUBTRACT CPP /YX /Yc /Yu

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\winpapi_console.c

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

# ADD CPP /Yu

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\WinPAPIShell.c

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

# ADD CPP /I "c:\papi\src\winpmc" /I "c:\papi\src\winpmc\sys" /I "..\winpmc" /I "..\winpmc\sys"

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

# ADD CPP /I "..\winpmc" /I "..\winpmc\sys"

!ENDIF 

# End Source File
# Begin Source File

SOURCE=.\WinPAPIShell.rc
# End Source File
# Begin Source File

SOURCE=..\..\tests\zero.c

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

# ADD CPP /I "c:/papi/src"
# SUBTRACT CPP /YX /Yc /Yu

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

# SUBTRACT CPP /YX /Yc /Yu

!ENDIF 

# End Source File
# End Group
# Begin Group "Header Files"

# PROP Default_Filter "h;hpp;hxx;hm;inl"
# Begin Source File

SOURCE=.\resource.h
# End Source File
# Begin Source File

SOURCE=.\StdAfx.h
# End Source File
# Begin Source File

SOURCE=..\..\tests\test_utils.h
# End Source File
# Begin Source File

SOURCE=.\WinPAPI_protos.h
# End Source File
# Begin Source File

SOURCE=.\WinPAPIShell.h
# End Source File
# End Group
# Begin Group "Resource Files"

# PROP Default_Filter "ico;cur;bmp;dlg;rc2;rct;bin;rgs;gif;jpg;jpeg;jpe"
# Begin Source File

SOURCE=.\bitmap3.bmp
# End Source File
# Begin Source File

SOURCE=.\Papismall.ico
# End Source File
# Begin Source File

SOURCE=.\WinPAPIShell.ico
# End Source File
# End Group
# Begin Group "Libraries"

# PROP Default_Filter "*.lib"
# Begin Source File

SOURCE=..\substrate\Debug\WinPAPI.lib

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

# PROP Exclude_From_Build 1

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

!ENDIF 

# End Source File
# Begin Source File

SOURCE=..\substrate\Release\WinPAPI.lib

!IF  "$(CFG)" == "WinPAPIShell - Win32 Release"

!ELSEIF  "$(CFG)" == "WinPAPIShell - Win32 Debug"

# PROP Exclude_From_Build 1

!ENDIF 

# End Source File
# End Group
# End Target
# End Project
