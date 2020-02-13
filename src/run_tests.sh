#!/bin/sh

# File:    papi.c
# Author:  Philip Mucci
#          mucci@cs.utk.edu
# Mods:    Kevin London
#          london@cs.utk.edu
#          Philip Mucci
#          mucci@cs.utk.edu

# if make sure that the tests are built
if [ "x$BUILD" != "x" ]; then
    cd testlib; make; cd ..
    cd validation_tests; make; cd ..
    cd ctests; make; cd ..
    cd ftests; make; cd ..
    for comp in `ls components/*/tests` ; do \
	cd components/$$comp/tests ; make; cd ../../.. ;
    done
fi

AIXTHREAD_SCOPE=S
export AIXTHREAD_SCOPE
if [ "X$1" = "X-v" ]; then
  shift ; TESTS_QUIET=""
else
# This should never have been an argument, but an environment variable!
  TESTS_QUIET="TESTS_QUIET"
  export TESTS_QUIET
fi

# Disable high-level output
if [ "x$TESTS_QUIET" != "xTESTS_QUIET" ] ; then
  export PAPI_REPORT=1
fi

if [ "x$VALGRIND" != "x" ]; then
  VALGRIND="valgrind --leak-check=full";
fi

VTESTS=`find validation_tests/* -prune -perm -u+x -type f ! -name "*.[c|h]"`;
#CTESTS=`find ctests -maxdepth 1 -perm -u+x -type f`;
CTESTS=`find ctests/* -prune -perm -u+x -type f ! -name "*.[c|h]"`;
FTESTS=`find ftests -perm -u+x -type f ! -name "*.[c|h|F]"`;
COMPTESTS=`find components/*/tests -perm -u+x -type f ! \( -name "*.[c|h]" -o -name "*.cu" \)`;
#EXCLUDE=`grep --regexp=^# --invert-match run_tests_exclude.txt`
EXCLUDE=`grep -v -e '^#\|^$' run_tests_exclude.txt`


ALLTESTS="$VTESTS $CTESTS $FTESTS $COMPTESTS";

PATH=./ctests:$PATH
export PATH

echo "Platform:"
uname -a

echo "Date:"
date

echo ""
if [ -r /proc/cpuinfo ]; then
   echo "Cpuinfo:"
   # only print info on first processor on x86
   sed '/^$/q' /proc/cpuinfo
fi

echo ""
if [ "x$VALGRIND" != "x" ]; then
  echo "The following test cases will be run using valgrind:";
else
  echo "The following test cases will be run:";
fi
echo ""

MATCH=0
LIST=""
for i in $ALLTESTS;
do
  for xtest in $EXCLUDE;
  do
    if [ "$i" = "$xtest" ]; then
      MATCH=1
      break
    fi;
  done
  if [ $MATCH -ne 1 ]; then
	LIST="$LIST $i"
  fi;
  MATCH=0
done
echo $LIST
echo ""

echo ""
echo "The following test cases will NOT be run:";
echo $EXCLUDE;

echo "";
echo "Running Tests";
echo ""

if [ "$LD_LIBRARY_PATH" = "" ]; then
  LD_LIBRARY_PATH=.:./libpfm-3.y/lib
else
  LD_LIBRARY_PATH=.:./libpfm-3.y/lib:"$LD_LIBRARY_PATH"
fi
export LD_LIBRARY_PATH
if [ "$LIBPATH" = "" ]; then
  LIBPATH=.:./libpfm-3.y/lib
else
  LIBPATH=.:./libpfm-3.y/lib:"$LIBPATH"
fi
export LIBPATH

echo ""
echo "Running Event Validation Tests";
echo ""

for i in $VTESTS;
do
  for xtest in $EXCLUDE;
  do
    if [ "$i" = "$xtest" ]; then
      MATCH=1
      break
    fi;
  done
  if [ $MATCH -ne 1 ]; then
    if [ -x $i ]; then
      RAN="$i $RAN"
      printf "Running %-50s %s" $i:
      $VALGRIND ./$i $TESTS_QUIET
      
      #delete output folder for high-level tests
      case "$i" in
        *"_hl"*) rm -r papi_hl_output ;;
      esac

    fi;
  fi;
  MATCH=0
done

echo ""
echo "Running C Tests";
echo ""

for i in $CTESTS;
do
  for xtest in $EXCLUDE;
  do
    if [ "$i" = "$xtest" ]; then
      MATCH=1
      break
    fi;
  done
  if [ $MATCH -ne 1 ]; then
    if [ -x $i ]; then
      RAN="$i $RAN"
      printf "Running %-50s %s" $i:
      $VALGRIND ./$i $TESTS_QUIET

      #delete output folder for high-level tests
      case "$i" in
        *"_hl"*) rm -r papi_hl_output ;;
      esac

    fi;
  fi;
  MATCH=0
done

echo ""
echo "Running Fortran Tests";
echo ""

for i in $FTESTS;
do
  for xtest in $EXCLUDE;
  do
    if [ "$i" = "$xtest" ]; then
      MATCH=1
      break
    fi;
  done
  if [ $MATCH -ne 1 ]; then
    if [ -x $i ]; then
      RAN="$i $RAN"
      printf "Running $i:\n"
      $VALGRIND ./$i $TESTS_QUIET

      #delete output folder for high-level tests
      case "$i" in
        *"_hl"*) rm -r papi_hl_output ;;
      esac

    fi;
  fi;
  MATCH=0
done

echo "";
echo "Running Component Tests";
echo ""

for i in $COMPTESTS;
do
  for xtest in $EXCLUDE;
  do
    if [ "$i" = "$xtest" ]; then
      MATCH=1
      break
    fi;
  done
  if [ $MATCH -ne 1 ]; then
    if [ -x $i ]; then
	RAN="$i $RAN"
    printf "Running $i:\n";
    printf "%-59s" ""
    $VALGRIND ./$i $TESTS_QUIET
    fi;
  fi;
  MATCH=0
done

if [ "$RAN" = "" ]; then 
	echo "FAILED to run any tests. (you can safely ignore this if this was expected behavior)"
fi;
