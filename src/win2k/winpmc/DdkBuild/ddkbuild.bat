@echo off
rem $Header$
set version=3.5.6
@echo DDBUILD VERSION %version% FREEWARE FROM HOLLIS TECHNOLOGY SOLUTIONS
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

rem next test, some shells is different
setlocal enabledelayedexpansion 
set VAR=before
if "%VAR%" == "before" (
     set VAR=after
     if "!VAR!" == "after" (
        @echo W2K or later system
        call :ddkbuild %*
      ) else (
        @echo nt4 or earlier system
        call :nt4ddkbuild %*
      )
)

goto :EOF

rem
rem
rem the latest and greatest ddkbuild
rem
rem
:ddkbuild
if "%1" EQU "-debug" shift

@echo %scriptDebug%

set w2kflag=
set chk=checked
set fre=free
set xp64=
set xp2k=0


if "%1" EQU ""     goto usage
if /I %1 EQU -h  goto usage
if /I %1 EQU /?    goto usage
if /I %1 EQU -help goto usage

if /I "%1" EQU "-W2K64" (
    set w2kflag=W2K64
    if "%W2K64BASE%" EQU "" goto NoW2k64Base
    set BASEDIR=%W2K64BASE%
    shift
	goto buildswitchdone
) 

if /I "%1" EQU "-W2K" (
    set w2kflag=W2K
    shift
    if "%W2KBASE%" EQU "" goto NoW2kBase
    set BASEDIR=%W2KBASE%
	goto buildswitchdone
)

if /I "%1" EQU "-XP" (
    set w2kflag=W2K
    shift
    if "%XPBASE%" EQU "" goto NoXPBase    
    set BASEDIR=%XPBASE%
    set chk=chk
    set fre=fre
	goto buildswitchdone
) 

if /I "%1" EQU "-XP64" (
    set w2kflag=W2K
    shift
    if "%XPBASE%" EQU "" goto NoXPBase    
    set BASEDIR=%XPBASE%
    set chk=chk
    set fre=fre
    set xp64=64
	goto buildswitchdone
) 

if /I "%1" EQU "-XPW2K" (
    set w2kflag=W2K
    set xp2k=1
    shift
    if "%XPBASE%" EQU "" goto NoXPBase    
    set BASEDIR=%XPBASE%
    set chk=chk
    set fre=fre
)

:buildswitchdone

if "%BASEDIR%" EQU "" goto ErrNoBASEDIR

set path=%BASEDIR%\bin;%path%

echo PATH: %PATH%

set mode=

if /I "%1" EQU "free" (

    set mode=%fre%

) else if /I "%1" EQU "checked" (

    set mode=%chk%

)

if "%mode%" EQU "" goto ErrBadMode

if "%2" EQU "" goto ErrNoDir

if not exist %2 goto ErrNoDir

set batfile=%BASEDIR%\bin\setenv.bat
if "%xp2k%" EQU "1" set batfile=%BASEDIR%\bin\w2k\set2k.bat

pushd .
if "%w2kflag%" NEQ "" (

	call %batfile% %BASEDIR% %mode% %xp64%

) else (

    call %BASEDIR%\bin\setenv.bat %BASEDIR% %mode%
)
popd

@echo %scriptDebug%

set mpFlag=-M

if "%BUILD_ALT_DIR%" EQU "" goto NT4

rem win2k sets this!
set W2kEXT=%BUILD_ALT_DIR%

if "%xp2k%" EQU "1" (

    set W2kEXT=%BUILD_ALT_DIR%_w2k

)

set mpFlag=-MI

:NT4

if "%NUMBER_OF_PROCESSORS%"=="" set mpFlag=
if "%NUMBER_OF_PROCESSORS%" EQU "1" set mpFlag=

@echo build in directory %2 with arguments %3 (basedir %BASEDIR%)

cd /d %2
set bflags=-e
set bscFlags=""

if "%3" NEQ "" (

    if "%3"  EQU  "/a" (

        set bscFlags=/n
        set bflags=-cfe 

    ) else (

        set bscFlags=/n

        set bflags=%3 -e
    )
)


if EXIST build%W2kEXT%.err  erase build%W2kEXT%.err
if EXIST build%W2kEXT%.wrn  erase build%W2kEXT%.wrn
if EXIST build%W2kEXT%.log  erase build%W2kEXT%.log


@echo run build %bflags% %mpFlag% for %mode% version in %2
pushd .
build  %bflags% %mpFlag%
popd

@echo %scriptDebug%

rem assume that the onscreen errors are complete!

@echo =============== build warnings ======================
if exist build%W2kEXT%.log findstr "warning.*[CLU][0-9]*" build%W2kEXT%.log

@echo. 
@echo. 
@echo build complete

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

:ErrnoDir
@echo Error: second parameter must be a valid directory

:usage
@echo usage: ddkbuild [-debug] "[-W2K|-W2K64|-XP|-XP64|-XPW2K]" "checked | free" "directory-to-build" [flags] 
@echo.
@echo        -debug     turns on script echoing for debug purposes
@echo.
@echo        -W2K       indicates development system uses W2KBASE environment variable
@echo                    to locate the win2000 ddk, otherwise BASEDIR is used (optional.)
@echo.
@echo        -W2K64     indicates development sytsem uses W2K64BASE environment variable
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
@echo        checked    indicates a checked build.
@echo.
@echo        free       indicates a free build (must choose one or the other of free or checked.)
@echo.
@echo        directory  path to build directory, try . (cwd).
@echo.
@echo        flags      any random flags or arguments you think should be passed to build (note that the
@echo                    visual studio /a for clean build is translated to the equivalent build flag.)
@echo.        
@echo         ex: ddkbuild -XP checked . 
@echo.    

rem goto end

:wrongplatform
@echo Sorry: NT4/W2K/XP only! 

:end

@echo ddkbuild complete

goto :EOF


rem
rem ===============================================================
rem ======================== SOME SHELLS DON'T LIKE if cond ( exp ) 
rem ===============================================================
rem
rem start the old ddkbuild
rem
:nt4ddkbuild
if "%1" EQU "-debug" shift
set w2kddk=0

rem allow for win2k ddk concurrent with nt4.0 ddk - use W2KBASE as a clue
rem only look at W2KBASE if %1 == -W2K otherwise just use BASEDIR

if /I "%1" NEQ "-W2K" goto NoW2kBase4

set w2kddk=1

shift

if "%W2KBASE%"=="" goto NoW2kBase4

set BASEDIR=%W2KBASE%

:NoW2kBase4
 
if "%BASEDIR%"=="" goto ErrNoBASEDIR4

set path=%BASEDIR%\bin;%path%

echo PATH: %PATH%

set mode=
for %%f in (free FREE checked CHECKED) do if %%f == %1 set mode=%1
if %mode%=="" goto ErrBadMode4

if "%2" == "" goto ErrNoDir4

if not exist %2 goto ErrNoDir4

pushd .
call %BASEDIR%\bin\setenv.bat %BASEDIR% %mode% "%MSDEV%"
popd

@echo %scriptDebug%

set mpFlag=-M

if "%BUILD_ALT_DIR%"=="" goto NT4

rem win2k sets this!
set W2kEXT=%BUILD_ALT_DIR%

set mpFlag=-MI

:NT4

if "%NUMBER_OF_PROCESSORS%"=="" set mpFlag=
if "%NUMBER_OF_PROCESSORS%"=="1" set mpFlag=

@echo build in directory %2 with arguments %3 (basedir %BASEDIR%)

cd %2
set bflags=-e
set bscFlags=""

if "%3" == "" goto done4

if "%3" == "/a" goto rebuildall4

set bscFlags=/n

set bflags=%3 -e

goto done4

:rebuildall4

set bscFlags=/n
set bflags=-cfe

:done4

if EXIST build%W2kEXT%.err  erase build%W2kEXT%.err
if EXIST build%W2kEXT%.wrn  erase build%W2kEXT%.wrn
if EXIST build%W2kEXT%.log  erase build%W2kEXT%.log


@echo run build %bflags% %mpFlag% for %mode% version in %2
pushd .
build  %bflags% %mpFlag%
popd

@echo %scriptDebug%

rem assume that the onscreen errors are complete!

@echo =============== build warnings ======================
if exist build%W2kEXT%.log findstr "warning.*[CLU][0-9]*" build%W2kEXT%.log

@echo. 
@echo. 
@echo build complete

@echo building browse information files

@echo off

if EXIST buildbrowse.cmd goto doBrowsescript4

set sbrlist=sbrList.txt

if not EXIST sbrList%CPU%.txt goto sbrDefault4

set sbrlist=sbrList%CPU%.txt

:sbrDefault4

if not EXIST %sbrlist% goto end4

if %bscFlags% == "" goto noBscFlags4

bscmake %bscFlags% @%sbrlist%

goto end4

:noBscFlags4

bscmake @%sbrlist%

goto end4

:doBrowsescript4

call buildBrowse %mode%

goto end4

:ErrBadMode4
@echo error: first param must be "checked" or "free"
goto usage4

:ErrNoBASEDIR4

if %w2kddk% == 1 goto forgotW2kBase4

@echo error: BASEDIR environment variable not set, reinstall DDK!

goto usage4

:forgotW2kBase4

@echo error: W2KBASE environment variable not set!

goto usage4

:ErrnoDir4
@echo Error: second parameter must be a valid directory

:usage4
@echo usage: ddkbuild [-debug] [-W2K] "checked | free" "directory-to-build" [flags] 
@echo.
@echo        -debug     turns on script echoing for debug purposes
@echo.
@echo        -W2K       indicates development system uses W2KBASE environment variable
@echo                   to locate the win2000 ddk, otherwise BASEDIR is used (optional)
@echo         checked   indicates a checked build
@echo         free      indicates a free build (must choose one or the other)
@echo         directory path to build directory, try . (cwd)
@echo         flags     any random flags you think should be passed to build (try /a for clean)
@echo.        
@echo         ex: ddkbuild checked . 
@echo.    

rem goto end4

:end4

@echo ddkbuild complete

goto :EOF