# .cshrc setup
# Created 1/23/97

#setenv OPENWINHOME /usr/openwin

###### Determine my  architecture type #####
## and hostname. Fill environment variables MYARCH, MYHOST

setenv MYARCH `uname`
set HOSTNAME_ARGS = ""
switch ("$MYARCH")
	case Linux:
		set HOSTNAME_ARGS = "-s"
	breaksw
endsw

if ( -x /usr/bin/hostname ) then
	set HOSTNAME = /usr/bin/hostname
else if ( -x /usr/bsd/hostname ) then
	set HOSTNAME = /usr/bsd/hostname
else if ( -x /sbin/hostname ) then
	set HOSTNAME = /sbin/hostname
else if ( -x /usr/ucb/hostname ) then
	set HOSTNAME = /usr/ucb/hostname
else if ( -x /bin/hostname ) then
	set HOSTNAME = /bin/hostname
else
	set HOSTNAME = echo -n UNKNOWN
endif 

setenv MYHOST `$HOSTNAME $HOSTNAME_ARGS`

####### Set TERMINAL Variables  #########

if ( $?TERM ) then
	set termtype = "$TERM"
else
	set termtype = "vt100"
endif

######	Set shell ##########
if ( -x /bin/tcsh ) then
	set	shell = /bin/tcsh
	setenv	SHELL /bin/tcsh
else
	set	shell = /bin/csh
	setenv	SHELL /bin/csh
endif

###### Set up Usable Path Information #########
set  minpath = ( . ~/bin ~/lib  /bin /usr/bin /sbin /usr/sbin /etc /usr/etc /usr/local /usr/local/bin /usr/local/lib /usr/ucb /usr/bsd /usr/local/sfw/bin )

#set  xpath   = ( /usr/bin/X11 /usr/openwin/bin /usr/local/X11/bin )
set  xpath   = ( /usr/bin/X11 /usr/local/X11/bin /usr/local/X11R6.3/bin )

set manpath = ( /usr/man /usr/local/man /usr/share/man /usr/X11/man  /usr/local/X11/man /src/icl/admin/man /flannel/homes/browne/Pablo/man /flannel/homes/browne/Pablo/Man)

set optpath = ( /opt /opt/bin /opt/wabi/bin )


switch ( "$MYARCH" )
	case AIX:
		set  path = ( $minpath /usr/lpp/X11/bin $xpath )
		breaksw
	case SunOS:
		set  path = (/usr/local/SUNWspro/bin /usr/local/java1.4.1/bin /usr/ucb $minpath /usr/lang $xpath /usr/local/gnu/bin /mdx/usr/local/matlab_r12.1/bin /usr/local/mpich/bin)
		set manpath = ($manpath /usr/lang/man /usr/local/gnu/man /usr/local/mpich/man)
		setenv CLASSPATH ~min/tij/javacode/:.:..:/sunshine/homes/bvz/cs594
		breaksw

	case Linux:
		set path = ( /usr/local/bin /usr/local/gcc-3.3.2/bin $minpath $optpath $xpath /usr/local/java1.4/bin /usr/local/mpich/bin /usr/local/globus/bin /usr/local/totalview/bin ~min/tau/i386_linux/bin)
		set manpath = ($manpath /opt/man /opt/wabi/man /usr/local/mpich/man /usr/local/globus/man ~min/papi/man)
		setenv CLASSPATH ~min/java/postgresql.jar:/linen/homes/min/java/xerces-2_4_0/xmlParserAPIs.jar:/linen/homes/min/java/xerces-2_4_0/xercesImpl.jar:/linen/homes/min/xerces-2_4_0/java/xml-apis.jar:/linen/homes/min/java/xerces-2_4_0/xercesSamples.jar:.
		setenv GLOBUS_LOCATION /usr/local/globus
		setenv LD_LIBRARY_PATH /usr/lib:/lib:/usr/local/gcc-3.3.2/lib
		breaksw
	case IRIX64:
		set path = ( $minpath $optpath $xpath /usr/java//bin)
		breaksw
	default:
		set  path = ($minpath $xpath)
		breaksw
endsw

#######   Set up path for TeX     ######

set path = ( /usr/local/teTeX/bin $path)


#######   Set up path for mpich     ######

#setenv	MPI_ROOT /paper/homes/browne/mpich
#set path = ( $path $MPI_ROOT/bin )
#set manpath =  ($manpath $MPI_ROOT/man)

#######	Set environment variables ###### 

setenv	EDITOR		'vi'
setenv	LESS  		'M'
setenv	MORE  		'-d'
setenv	PAGER  		'more'

###### Setup the default Printer ###### 
setenv PRINTER 'cl304'  


####### Set some predefined Shell Variables ########

set	fignore	=	( .o .aux .bbl .blg .dvi .glo .idx .lof .log .lot .ps \
			.toc .bak .dlog)
set	filec
set	history=30
set	ignoreeof
set	noclobber
set	notify
set	savehist=30
set	time=15
set filec


####### Set Limits ########
 limit	cputime		unlimit
# limit	coredumpsize 	0


###### Define how to and set the PROMPT #####

set prompt="$MYHOST{`pwd`}% "

########## ALIASES ########### 

alias	back		'set back=$old; set old=$cwd; cd $back; unset back; pwd'
alias	bc 		'bc -l'
alias	bye		'logout'
alias	cd		'set old=$cwd; chdir \!*; pwd'
alias	compressdir	'find \!^ -type f -exec compress {} \;'
alias   cp              '/bin/cp -i'
alias	displayto	'setenv DISPLAY \!^.epm.ornl.gov:0.0'
alias	dvispool 	'dvi2ps \!* | lpr'
alias	findfile 	'find . -name \*\!^\* -print'
alias	grep		'grep -i'
alias	h		'history'
alias	hide		'chmod go-rw'
alias	iama		'set noglob ; eval `tset -s -Q \!^ \!*`'
alias	jobs		'jobs -l'
alias	lo 		'logout'
alias	lower 		'/bin/mv -f \!^ \!^.upper; tr A-Z a-z < \!^.upper > \!^ ; compress -f \!^.upper'
alias	ls 		'/bin/ls -F'
alias	lsall 		'/bin/ls -lAF \!* |more'
alias	lsbig 		'/bin/ls -lsAF  \!* |sort +4nr |more'
alias	lsdirs 		'/bin/ls -lAF \!* | /bin/grep / |more'
alias	lsdirsall 	'find . -type d -print |more'
alias	lslast 		'/bin/ls -ltAF \!* |more'
alias	lslong 		'/bin/ls -lF \!* |more'
alias	mv 		'/bin/mv -i'
alias	phone 		'fgrep -i \!^ /home/sun1/u0/darland/numbers/phone; fgrep -i \!^ ~/.phone'
alias	preview 	'xdvi \!* &'
alias	protect 	'chmod a-w'
alias	psgrep		'ps aux | sed -n -e "/sed -n -e /d\\
                	/\!$/p\\
                	/TIME COMMAND/p"'
alias	ptroff		"cat \!* | rsh seq ' psdipic | psditbl | psdieqn | psditroff -ms ' &"
alias	ptroffmemo	"cat \!* | rsh seq ' psdipic | psditbl | psdieqn | psditroff -mqo ' &"
# alias	rm		'/bin/rm -i'
alias	rolo		'/usr/local/rolo \!* -u ~rolo'
alias	save 		'cp \!^ \!^.save; compress -f \!^.save'
alias	savedate 	'set day=`date`;set ext=$day[2]_$day[3]_$day[6]; cp \!^ \!^.$ext;compress -f \!^.$ext'
alias	spell 		'ispell -t'
alias	ts		'set noglob ; eval `tset -s -Q -m dca:vt100 \!*`'
alias	uncompressdir	'find \!^ -type f -exec uncompress {} \;'
alias	unhide 		'chmod go+r'
alias	unprotect 	'chmod u+w'
alias	upper 		'/bin/mv -f \!^ \!^.lower; tr a-z A-Z < \!^.lower > \!^ ; compress -f \!^.lower'
alias	whosname	'whos "/full \!^" | less'
alias	whosphone	'whos "/full/phone=\!^" | less'
alias	whosuid		'whos "/full/uid=\!^" | less'
alias	xvi	 	'xterm -geometry 92x40 -e vi \!* &'
alias	zless 		'zcat \!^ | less'
alias	zmore 		'zcat \!^ | more'

###### Added for PVM support #######
setenv PVM_ROOT /paper/homes/browne/pvm/pvm3
if (! $?PVM_ROOT) then
	if (-d ~/pvm3) then
		setenv PVM_ROOT ~/pvm3
	else
		echo PVM_ROOT not defined
		echo To use PVM, define PVM_ROOT and rerun your .cshrc
	endif
endif

#if ($?PVM_ROOT) then
#        setenv PVM_EXPORT DISPLAY:CPDPATH
#		setenv PVM_ARCH `$PVM_ROOT/lib/pvmgetarch`
#        set manpath =  ($manpath $PVM_ROOT/man)
#        setenv XPVM_ROOT /paper/homes/browne/pvm/xpvm
#        setenv TCL_LIBRARY $XPVM_ROOT/src/tcl
#        setenv TK_LIBRARY $XPVM_ROOT/src/tcl 
 
#
# uncomment the following line if you want the PVM executable directory
# to be added to your shell path.
#
#	set path=($path $PVM_ROOT/bin/$PVM_ARCH)
#	set path=($path $PVM_ROOT/lib)
#	set path=($path $PVM_ROOT/lib/$PVM_ARCH)

 
#endif

#if ($?PVM_ROOT) then
#  switch ("$PVM_ARCH")
#	case SUNMP:
#		set MPIARCH = "solaris"
#	breaksw
#	case SUN4SOL2:
#		set MPIARCH = "solaris"
#	breaksw
#        case SGI64:
#                set MPIARCH = "IRIX64"
#        breaksw
#        case SGI5:
#                set MPIARCH = "IRIX"
#        breaksw
#        case SUN4:
#                set MPIARCH = "sun4"
#        breaksw
#        case RS6K:
#                set MPIARCH = "rs6000"
#        breaksw
#endsw

#set path = ($MPI_ROOT/lib/$MPIARCH/ch_p4 $path)

#switch ("$MPIARCH")
#       case solaris:
#               set path = (/opt/SUNWspro/bin $path /usr/ccs/bin)
#       breaksw
#endsw

##### Set the MANPATH environment now #####
setenv MANPATH `echo $manpath | sed 's/ /:/g'`

#setenv CLASSPATH /home/mcmahan/java/classes/swing.jar:/home/mcmahan/java/classes/xml4j_1_1_9.jar

setenv CVSROOT /cvs/homes/papi
setenv MPIR_HOME /usr/local/mpich
#setenv MPI_ARCH `$MPIR_HOME/bin/tarch`
#setenv MPI_LIB $MPIR_HOME/lib/$MPI_ARCH/ch_p4
#set path = ($path $MPIR_HOME/bin $MPI_LIB)
#setenv MANPATH $MPIR_HOME/man:$MANPATH
