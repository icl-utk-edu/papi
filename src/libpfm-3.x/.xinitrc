#!/bin/sh

# merge in defaults and keymaps

USERRESOURCES=$HOME/.Xresources
USERMODMAP=$HOME/.Xmodmap

if [ -f $USERRESOURCES ]; then
    xrdb -merge $USERRESOURCES
fi
if [ -f $USERMODMAP ]; then
    xmodmap $USERMODMAP
fi

# biff and clock
xbiff -fg white -bg slategrey -geometry 100x100-95+0 &
xclock -geometry 90x100-0+0 -update 60 -fg white -bg slategray -hl white -hd white &

# console window
xterm +sb -fg white -bg black -C -geometry 45x6-200+8 -fn 6x13 -name console &

xsetroot -solid slategrey

# run fvwm2
exec fvwm2
