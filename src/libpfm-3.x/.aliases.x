#!/bin/csh -fxv
#################################################################
#
# this file should be $HOME/.aliases.x 
# Aliases for interactive shells in x windows.
# writen by Don Sparkman 
#

if("$term" != "xterm" ) exit 

#since this is an xterm and xbiff turn off mail notification here
if($?mail) unset mail

#make sure the variables are set 

if(($?windname == 0 ) && ($?WINDNAME)) then 
   set windname = ($WINDNAME)
else
   set windname = (`hostname`)
endif  
setenv  WINDNAME   `hostname`

###########alias for titlebar manipulation 

alias _titlebar  'echo -n "]2;\!*"' 
alias titleiconbar  'echo -n "];\!*"' 
alias iconbar 'echo -n "]1;\!*"' 
alias chtb 'echo -n "]2; $windname `tpwd` " '
   
###########set titlebar for various commands 

alias vi '_titlebar  $windname  VI  \!* ; /usr/ucb/vi \!* ; chtb ;echo "" ' 
alias popd 'popd ;chtb'
alias pushd 'pushd \!*  ;chtb'

alias tcsh  'set oldwindname = $WINDNAME;setenv WINDNAME $windname;/usr/local/bin/tcsh \!* ; setenv WINDNAME $oldwindname;unset oldwindname;chtb'

alias csh  'set oldwindname = $WINDNAME;setenv WINDNAME $windname; /bin/csh \!* ; setenv WINDNAME $oldwindname;unset oldwindname;chtb'

#alias rlogin '\rlogin \!*;chtb;titleiconbar $HOST'
#alias rsh '\rsh \!*;chtb;titleiconbar $HOST'


chtb 
iconbar $windname 

#for tcsh users 
#this allows each window to have its own history file
#this is a useless variable to csh 

# it must be uncommented before it can be used in tcsh 
# set histfile = ~/.$windname.history

