@echo off
rem $Header$
set version=3.9.27
@echo DDKBUILD VERSION %version% FREEWARE FROM HOLLIS TECHNOLOGY SOLUTIONS
@echo Comments? Suggestions? info@hollistech.com

set scriptDebug=off
if "%1" NEQ "-debug" goto nodebug
set scriptDebug=on
shift

:nodebug
@echo %scriptDebug%

VERIFY OTHER 2>nul
setlocal ENABLEEXTENSIONS
if ERRORLEVEL 1 goto wrongplatform

rem some shells is different
set foo=dog
if %foo% EQU cat (
    set foo=rat 
) else (
    set foo=cat
)
if %foo% EQU dog goto :nt4ddkbuild   

if /I "%1" EQU "/?"    goto usage   

rem next test, some shells is different
setlocal enabledelayedexpansion 
set VAR=before
if "%VAR%" == "before" (
     set VAR=after
     if "!VAR!" == "after" (
        rem @echo W2K or later system
        call :ddkbuild %*
      ) else (
        @echo nt4 or earlier system
        call :nt4ddkbuild %*
      )
)

goto :EOF

rem ================
rem the latest and greatest ddkbuild
rem ================

:ddkbuild

if "%1" == "-debug" shift

set w2kflag=
rem ================
rem NOTE: w2kflag appears to be somewhat obsolete,
rem at best it is either null or non-null.
rem buildbsc may use its value, but the setting of 
rem that value is inconsistent.
rem ================

set chk=checked
set fre=free
set xp64=
set xp2k=0
set wnet=
set wlh=
set ddk=UNKNOWN
set mode=


if "%1" EQU ""     goto usage
if /I %1 EQU -h  goto usage
if /I %1 EQU -help goto usage

rem test for all known ddk variations

rem =========================
rem W2K DDK Support (Obsolete)
rem =========================

rem 64bit w2k
if /I "%1" EQU "-W2K64" (
    set w2kflag=W2K64
    if "%W2K64BASE%" EQU "" goto NoW2k64Base
    set BASEDIR=%W2K64BASE%
    set ddk=W2K
    shift
    goto buildswitchdone
) 

rem regular w2k
if /I "%1" EQU "-W2K" (
    set w2kflag=W2K
    shift
    if "%W2KBASE%" EQU "" goto NoW2kBase
    set BASEDIR=%W2KBASE%
    set ddk=W2K
    goto buildswitchdone
)

rem =========================
rem XP DDK Support (Obsolete)
rem =========================

rem regular xp
if /I "%1" EQU "-XP" (
    set w2kflag=W2K
    shift
    if "%XPBASE%" EQU "" goto NoXPBase    
    set BASEDIR=%XPBASE%
    set chk=chk
    set fre=fre
    set ddk=XP
    goto buildswitchdone
) 

rem 64bit xp
if /I "%1" EQU "-XP64" (
    set w2kflag=W2K
    shift
    if "%XPBASE%" EQU "" goto NoXPBase    
    set BASEDIR=%XPBASE%
    set chk=chk
    set fre=fre
    set xp64=64
    set ddk=XP
    goto buildswitchdone
) 

rem w2k build/xp ddk
if /I "%1" EQU "-XPW2K" (
    set w2kflag=W2K
    set xp2k=1
    shift
    if "%XPBASE%" EQU "" goto NoXPBase    
    set BASEDIR=%XPBASE%
    set chk=checked
    set fre=free
    set ddk=XP
    goto buildswitchdone
)

rem =========================
rem NET DDK Support
rem =========================

rem .net ddk .net build
if /I "%1" EQU "-WNET" (
    set w2kflag=W2K
    shift
    if "%WNETBASE%" EQU "" goto NoWNBase    
    set BASEDIR=%WNETBASE%
    set chk=chk
    set fre=fre
    set wnet=wnet
    set ddk=NET
    goto buildswitchdone
)

rem .net ddk w2k build
if /I "%1" EQU "-WNETW2K" (
    set w2kflag=NET
    shift
    if "%WNETBASE%" EQU "" goto NoWNBase    
    set BASEDIR=%WNETBASE%
    set chk=chk
    set fre=free
    set wnet=w2k
    set ddk=NET
    goto buildswitchdone
)

rem .net ddk xp build
if /I "%1" EQU "-WNETXP" (
    set w2kflag=NET
    shift
    if "%WNETBASE%" EQU "" goto NoWNBase    
    set BASEDIR=%WNETBASE%
    set chk=chk
    set fre=fre
    set wnet=wxp
    set ddk=NET
    goto buildswitchdone
)

rem .net ddk IA64 build
if /I "%1" EQU "-WNET64" (
    set w2kflag=NET
    shift
    if "%WNETBASE%" EQU "" goto NoWNBase    
    set BASEDIR=%WNETBASE%
    set chk=chk
    set fre=fre
    set xp64=64
    set wnet=wnet
    set ddk=NET
    goto buildswitchdone
)

rem .net ddk AMD64 build
if /I "%1" EQU "-WNETA64" (
    set w2kflag=NET
    shift
    if "%WNETBASE%" EQU "" goto NoWNBase    
    set BASEDIR=%WNETBASE%
    set chk=chk
    set fre=fre
    set xp64=AMD64
    set wnet=wnet
    set ddk=NET
    goto buildswitchdone
)

rem ============================
rem LONGHORN DDK SUPPORT (BETA)
rem ============================

rem wlh ddk wlh build
if /I "%1" EQU "-WLH" (
    set w2kflag=W2K
    shift
    if "%WLHBASE%" EQU "" goto NoWLHBase    
    set BASEDIR=%WLHBASE%
    set chk=chk
    set fre=fre
    set wnet=wlh
    set ddk=WLH
    goto buildswitchdone
)

rem wlh ddk IA64 net build
if /I "%1" EQU "-WLH64" (
    set w2kflag=NET
    shift
    if "%WLHBASE%" EQU "" goto NoWLHBase    
    set BASEDIR=%WLHBASE%
    set chk=chk
    set fre=fre
    set xp64=64
    set wnet=wlh
    set ddk=WLH
    goto buildswitchdone
)

rem wlh ddk AMD64 net build
if /I "%1" EQU "-WLHA64" (
    set w2kflag=NET
    shift
    if "%WLHBASE%" EQU "" goto NoWLHBase    
    set BASEDIR=%WLHBASE%
    set chk=chk
    set fre=fre
    set xp64=AMD64
    set wnet=wlh
    set ddk=WLH
    goto buildswitchdone
)

rem wlh ddk w2k build
if /I "%1" EQU "-WLHW2K" (
    set w2kflag=NET
    shift
    if "%WLHBASE%" EQU "" goto NoWLHBase    
    set BASEDIR=%WLHBASE%
    set chk=chk
    set fre=free
    set wnet=w2k
    set ddk=WLH
    goto buildswitchdone
)

rem wlh ddk xp build
if /I "%1" EQU "-WLHXP" (
    set w2kflag=NET
    shift
    if "%WLHBASE%" EQU "" goto NoWLHBase    
    set BASEDIR=%WLHBASE%
    set chk=chk
    set fre=fre
    set wnet=wxp
    set ddk=WLH
    goto buildswitchdone
)

rem wlh ddk xp IA64 build
if /I "%1" EQU "-WLHXP64" (
    set w2kflag=NET
    shift
    if "%WLHBASE%" EQU "" goto NoWLHBase    
    set BASEDIR=%WLHBASE%
    set chk=chk
    set fre=fre
    set wnet=wxp
    set xp64=64
    set ddk=WLH
    goto buildswitchdone
)

rem wlh ddk .net build
if /I "%1" EQU "-WLHNET" (
    set w2kflag=W2K
    shift
    if "%WLHBASE%" EQU "" goto NoWLHBase    
    set BASEDIR=%WNETBASE%
    set chk=chk
    set fre=fre
    set wnet=wnet
    set ddk=WLH
    goto buildswitchdone
)

rem wlh ddk IA64 net build
if /I "%1" EQU "-WLHNET64" (
    set w2kflag=NET
    shift
    if "%WLHBASE%" EQU "" goto NoWLHBase    
    set BASEDIR=%WLHBASE%
    set chk=chk
    set fre=fre
    set xp64=64
    set wnet=wnet
    set ddk=WLH
    goto buildswitchdone
)

rem wlh ddk AMD64 net build
if /I "%1" EQU "-WLHNETA64" (
    set w2kflag=NET
    shift
    if "%WLHBASE%" EQU "" goto NoWLHBase    
    set BASEDIR=%WLHBASE%
    set chk=chk
    set fre=fre
    set xp64=AMD64
    set wnet=wnet
    set ddk=WLH
    goto buildswitchdone
)

:buildswitchdone

rem =====================
rem arg sanity tests
rem =====================

if "%w2kflag%" EQU "" goto BadTarget

if "%BASEDIR%" EQU "" goto ErrNoBASEDIR
set path=%BASEDIR%\bin;%path%
if "%scriptDebug%" EQU "on" @echo PATH: %PATH%

if /I "%1" EQU "free" (

    set mode=%fre%

) else if /I "%1" EQU "checked" (

    set mode=%chk%

)

if "%mode%" EQU "" goto ErrBadMode
shift

if "%1" EQU "" goto ErrNoDir
if not exist %1 goto ErrNoDir
set buildDir=%1
shift

set batfile=%BASEDIR%\bin\setenv.bat
if "%xp2k%" EQU "1" set batfile=%BASEDIR%\bin\w2k\set2k.bat

rem =====================
rem invoke the DDK's setenv script
rem =====================

pushd .
if "%w2kflag%" NEQ "" (

	call %batfile% %BASEDIR% %mode% %xp64% %wnet%

) else (

    call %BASEDIR%\bin\setenv.bat %BASEDIR% %mode%
)
popd

@echo %scriptDebug%

rem =====================
rem fixup the multiprocessor flag
rem so that Visual Studio doesn't get confused
rem =====================

set mpFlag=-M
if "%BUILD_ALT_DIR%" EQU "" goto NT4
set W2kEXT=%BUILD_ALT_DIR%
set mpFlag=-MI
:NT4
if "%NUMBER_OF_PROCESSORS%"=="" set mpFlag=
if "%NUMBER_OF_PROCESSORS%" EQU "1" set mpFlag=
set extraArgs=%~1

rem =====================
rem add any locally generated extra args to build command
rem =====================

cd /d %2
set bflags=-e
set bscFlags=""

if "%extraArgs%" NEQ "" (

    if "%extraArgs%"  EQU  "/a" (
        set bscFlags=/n
        set bflags=-cfe 

    ) else (
        set bscFlags=/n
        set bflags=%extraArgs% -e
    )
)
shift

rem ===================
rem add any remaining commandline arguments to extraArgs
rem ===================

set moreExtraArgs=%1 %2 %3 %4 %5 %6 %7 %8 %9


if EXIST build%W2kEXT%.err  erase build%W2kEXT%.err
if EXIST build%W2kEXT%.wrn  erase build%W2kEXT%.wrn
if EXIST build%W2kEXT%.log  erase build%W2kEXT%.log

@echo.
@echo DDKBUILD using %ddk% DDK in directory %buildDir% 
@echo  for %mode% version (basedir %BASEDIR% extension %W2kEXT%)
@echo  commandline: "build %bflags% %mpFlag% %moreExtraArgs%" 
@echo.
pushd .
pushd %buildDir%
build  %bflags% %mpFlag% %moreExtraArgs%
popd
popd

@echo %scriptDebug%

rem ===================
rem assume that the onscreen errors are complete!
rem ===================

@echo =============== build warnings ======================
if exist build%W2kEXT%.log findstr "warning.*[CLU][0-9]*" build%W2kEXT%.log

@echo. 
@echo. 
@echo build complete

rem ===================
rem BSCMAKE support
rem ===================

@echo building browse information files

@echo %scriptDebug%

if EXIST buildbrowse.cmd goto doBrowsescript
set sbrlist=sbrList.txt
if not EXIST sbrList%CPU%.txt goto sbrDefault
set sbrlist=sbrList%CPU%.txt

:sbrDefault
if not EXIST %sbrlist% goto end
if %bscFlags% == "" goto noBscFlags
bscmake %bscFlags% @%sbrlist%
goto end

rem ===================
rem error handlers
rem ===================

:noBscFlags
bscmake @%sbrlist%
goto end

:doBrowsescript
call buildBrowse %mode% %w2kflag%
goto end

:ErrBadMode
@echo error: first param must be "checked" or "free"
goto usage

:ErrNoBASEDIR
@echo error: BASEDIR environment variable not set, reinstall DDK!
goto usage

:NoW2kBase
@echo error: W2KBASE environment variable not set!
goto usage

:NoW2k64Base
@echo error: W2K64BASE environment variable not set!
goto usage

:NoXPBase
@echo error: XPBASE environment variable not set!
goto usage

:NoWNBase
@echo error: WNETBASE environment variable not set!
goto usage

:NoWLHBase
@echo error: WLHBASE environment variable not set!
goto usage

:ErrnoDir
@echo Error: second parameter must be a valid directory
goto usage

:BadTarget
@echo Error: invalid TARGET specified
goto usage

:usage
start http:\\www.hollistech.com\Resources\ddkbuild\ddkbuildhelp3_9.htm

@echo usage: ddkbuild [-debug] "TARGET" "checked | free" "directory-to-build" [flags] 
@echo.
@echo        -debug     turns on script echoing for debug purposes
@echo.
@echo        TARGET     can be any of the following combinations of DDK and target platform:
@echo.
@echo        -W2K       indicates development system uses W2KBASE environment variable
@echo                    to locate the win2000 ddk, otherwise BASEDIR is used (optional.)
@echo.
@echo        -W2K64     indicates development syatem uses W2K64BASE environment variable
@echo                    to locate the win2000 64 ddk, otherwise BASEDIR is used (optional.)
@echo.
@echo        -XP        indicates development system uses XPBASE environment variable
@echo                    to locate the XP ddk, otherwise BASEDIR is used (optional.)
@echo.
@echo        -XP64      indicates development system uses XPBASE environment variable
@echo                    to locate the XP ddk and builds IA64 binaries (optional.)
@echo.
@echo        -XPW2K     indicates development system uses the XPBASE environment variable
@echo                    to locate the XP ddk and builds W2K binaries (optional.)
@echo.
@echo        -WNET      indicates development system uses WNETBASE environment variable
@echo                    to locate the .Net ddk and builds .net binaries (optional.)
@echo.
@echo        -WNETW2K   indicates development system uses the WNETBASE environment variable
@echo                    to locate the .Net ddk and builds W2K binaries (optional.)
@echo.
@echo        -WNETXP    indicates development system uses WNETBASE environment variable
@echo                    to locate the .Net ddk and builds xp binaries (optional.)
@echo.
@echo        -WNET64    indicates development system uses WNETBASE environment variable
@echo                    to locate the .Net ddk and builds 64bit binaries (optional.)
@echo.
@echo        -WNETA64   indicates development system uses WNETBASE environment variable
@echo                    to locate the .Net ddk and builds AMD 64bit binaries (optional.)
@echo.
@echo        -WLH       indicates development system uses the WHLBASE environment variable
@echo                    to locate the Longhorn ddk and builds Longhorn binaries (optional.)
@echo.
@echo        -WLH64     indicates development system uses the WHLBASE environment variable
@echo                    to locate the Longhorn ddk and builds IA64 Longhorn binaries (optional.)
@echo.
@echo        -WLHA64    indicates development system uses the WHLBASE environment variable
@echo                    to locate the Longhorn ddk and builds AMD64 Longhorn binaries (optional.)
@echo.
rem @echo        -WLHW2K    indicates development system uses the WHLBASE environment variable
rem @echo                    to locate the Longhorn ddk and builds W2K binaries (optional.)
rem @echo.
rem @echo        -WLHXP     indicates development system uses the WHLBASE environment variable
rem @echo                    to locate the Longhorn ddk and builds XP binaries (optional.)
rem @echo.
rem @echo        -WLHXP64   indicates development system uses the WHLBASE environment variable
rem @echo                    to locate the Longhorn ddk and builds XP IA64 binaries (optional.)
rem @echo.
@echo        -WLHNET    indicates development system uses the WHLBASE environment variable
@echo                    to locate the Longhorn ddk and builds .net binaries (optional.)
@echo.
rem @echo        -WLHNET64  indicates development system uses the WHLBASE environment variable
rem @echo                    to locate the Longhorn ddk and builds IA64 .bet binaries (optional.)
rem @echo.
rem @echo        -WLHNETA64 indicates development system uses the WHLBASE environment variable
rem @echo                    to locate the Longhorn ddk and builds AMD64 .net binaries (optional.)
rem @echo.
@echo        checked    indicates a checked build.
@echo.
@echo        free       indicates a free build (must choose one or the other of free or checked.)
@echo.
@echo        directory  path to build directory, try . (cwd).
@echo.
@echo        flags      any random flags or arguments you think should be passed to build (note that the
@echo                    visual studio /a for clean build is translated to the equivalent build flag.)
@echo					 Note also that multiple arguments can be specified by using quotes to contain
@echo					 the set of arguments, as in "-Z foo blortz w2k xp"
@echo.        
@echo         ex: ddkbuild -XP checked . 
@echo.
@echo         NOTE: windows .net DDK versions supported must be build 3663 or later 
@echo.

goto :EOF

rem ======================
rem bad shell error handlers
rem ======================

:wrongplatform
@echo Sorry: NT4/W2K/XP/.net only! 
goto end

:nt4ddkbuild

@echo Sorry ddkbuild supports windows2000 or later platforms only
goto end

:end
