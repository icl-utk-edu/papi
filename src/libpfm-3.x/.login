##################################################################
#
#         .login file
#
#         Read in after the .cshrc file when you log in.
#         Not read in for subsequent shells.  For setting up
#         terminal and global environment characteristics.
#
##################################################################

#	Set terminal characteristics for remote terminals:
#	If TERM is a sun or vt100 or xterm, leave it alone.
#	Otherwise, query for the terminal type.
#	In any case, set TERMCAP to speed things up.

set noglob
if ("$term" == "sun"  ||  "$term" == "dtterm" || "$term" == "vt100"  ||  "$term" == "xterm") then
	eval `/usr/ucb/tset -sQ $term`
else
	eval `/usr/ucb/tset -sQ ?$TERM`
endif
unset noglob

#         general terminal characteristics

#stty -crterase
#stty -tabs
#stty crt
 stty erase '^H'
 stty werase '^?'
 stty kill '^U'

#         Commands to perform at login.
#         Note that exiting Xwindows leads automatically to logout.

#echo "!=<"     # turn off key click

biff y		# Notify me of mail right away.

if ("`tty`" != "/dev/console") exit

echo " "
echo -n "Xwindows? (^C to interrupt) "
sleep 2
xinit
logout
