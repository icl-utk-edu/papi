#!/usr/bin/perl

use warnings;
use strict;

# arguments: one of more report files
#
# Christian Mautner <christian * mautner . ca>, 2005-10-31
# Marc Schoechlin <ms * 256bit.org>, 2007-12-02
#
# This script is just a hack :-)
#
# This script is based loosely on the Generate_Graph set
# of scripts that come with iozone, but is a complete re-write
#
# The main reason to write this was the need to compare the behaviour of
# two or more different setups, for tuning filesystems or
# comparing different pieces of hardware.
#
# This script is in the public domain, too short and too trivial
# to deserve a copyright.
#
# Simply run iozone like, for example, ./iozone -a -g 4G > config1.out (if your machine has 4GB)
#
# and then run perl report.pl config1.out
# or get another report from another box into config2.out and run
# perl report.pl config1.out config2.out
# the look in the report_* directory for .png
#
# If you don't like png or the graphic size, search for "set terminal" in this file and put whatever gnuplot
# terminal you want. Note I've also noticed that gnuplot switched the set terminal png syntax
# a while back, you might need "set terminal png small size 900,700"
#
use Getopt::Long;

my $column;
my %columns;
my $datafile;
my @datafiles;
my $outdir;
my $report;
my $nooffset=0;
my @Reports;
my @split;
my $size3d; my $size2d;

# evaluate options
GetOptions(
    '3d=s'     => \$size3d,
    '2d=s'     => \$size2d,
    'nooffset' => \$nooffset
);

$size3d = "900,700" unless defined $size3d;
$size2d = "800,500" unless defined $size2d;


my $xoffset = "offset -7";
my $yoffset = "offset -3";

if ($nooffset == 1){
   $xoffset = ""; $yoffset = "";
}

print "\niozone_visualizer.pl : this script is distributed as public domain\n";
print "Christian Mautner <christian * mautner . ca>, 2005-10-31\n";
print "Marc Schoechlin <ms * 256bit.org>, 2007-12-02\n";


@Reports=@ARGV;

die "usage: $0 --3d=x,y -2d=x,y <iozone.out> [<iozone2.out>...]\n" if not @Reports or grep (m|^-|, @Reports);

die "report files must be in current directory" if grep (m|/|, @Reports);

print "Configured xtics-offset '$xoffset', configured ytics-offfset '$yoffset' (disable with --nooffset)\n";
print "Size 3d graphs : ".$size3d." (modify with '--3d=x,y')\n";
print "Size 2d graphs : ".$size2d." (modify with '--2d=x,y')\n";

#KB reclen write rewrite read reread read write read rewrite read fwrite frewrite fread freread
%columns=(
         'KB'        =>1,
         'reclen'    =>2,
         'write'     =>3,
         'rewrite'   =>4,
         'read'      =>5,
         'reread'    =>6,
         'randread'  =>7,
         'randwrite' =>8,
         'bkwdread'  =>9,
         'recrewrite'=>10,
         'strideread'=>11,
         'fwrite'    =>12,
         'frewrite'  =>13,
         'fread'     =>14,
         'freread'   =>15,
         );

#
# create output directory. the name is the concatenation
# of all report file names (minus the file extension, plus
# prefix report_)
#
$outdir="report_".join("_",map{/([^\.]+)(\..*)?/ && $1}(@Reports));

print STDERR "Output directory: $outdir ";

if ( -d $outdir )
{
    print STDERR "(removing old directory) ";
    system "rm -rf $outdir";
}

mkdir $outdir or die "cannot make directory $outdir";

print STDERR "done.\nPreparing data files...";

foreach $report (@Reports)
{
    open(I, $report) or die "cannot open $report for reading";
    $report=~/^([^\.]+)/;
    $datafile="$1.dat";
    push @datafiles, $datafile;
    open(O, ">$outdir/$datafile") or die "cannot open $outdir/$datafile for writing";
    open(O2, ">$outdir/2d-$datafile") or die "cannot open $outdir/$datafile for writing";

    my @sorted = sort { $columns{$a} <=> $columns{$b} } keys %columns;
    print O "# ".join(" ",@sorted)."\n";
    print O2 "# ".join(" ",@sorted)."\n";

    while(<I>)
    {
        next unless ( /^[\s\d]+$/ );
        @split = split();
        next unless ( @split == 15 );
        print O;
        print O2 if $split[1] == 16384 or $split[0] == $split[1];
    }
    close(I);
    close(O);
    close(O2);
}

print STDERR "done.\nGenerating graphs:";


open(HTML, ">$outdir/index.html") or die "cannot open $outdir/index.html for writing";

print HTML qq{<?xml version="1.0" encoding="iso-8859-1"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" lang="en" xml:lang="en">
<head>
<title>IOZone Statistics</title>
 <STYLE type="text/css">
.headline \{ font-family: Arial, Helvetica, sans-serif; font-size: 18px; color: 003300 ; font-weight: bold; text-decoration: none\}
 </STYLE>
</head>
<body>
<a name="top"></a>
<h1>IOZone Statistics</h1>
<table width="100%" summary="iozone stats">
<tr>
<td>
};

# Generate Menu
print HTML "<u><b>## Overview</b></u>\n<ul>\n";
foreach $column (keys %columns){
    print HTML '<li><b>'.uc($column).'</b> : '.
                   '<a href="#'.$column."\">3d</a>\n".
                   '<a href="#s2d-'.$column."\">2d</a></li>\n";
}
print HTML "</ul></td></tr>\n";
# Genereate 3d plots
foreach $column (keys %columns)
{
    print STDERR " $column";

    open(G, ">$outdir/$column.do") or die "cannot open $outdir/$column.do for writing";



    print G qq{
set title "Iozone performance: $column"
set grid lt 2 lw 1
set surface
set parametric
set xtics $xoffset
set ytics $yoffset
set logscale x 2
set logscale y 2
set autoscale z
#set xrange [2.**5:2.**24]
set xlabel "File size in KBytes" -2
set ylabel "Record size in Kbytes" 2
set zlabel "Kbytes/sec" 4,8
set style data lines
set dgrid3d 80,80,3
#set terminal png small picsize 900 700
set terminal png small size $size3d nocrop
set output "$column.png"
};

    print HTML qq{
      <tr>
       <td align="center">
         <h2><a name="$column"></a>3d-$column</h2><a href="#top">[top]</a><BR/>
         <img src="$column.png" alt="3d-$column"/><BR/>
       </td>
      </tr>
    };

    print G "splot ". join(", ", map{qq{"$_" using 1:2:$columns{$column} title "$_"}}(@datafiles));

    print G "\n";

    close G;

    open(G, ">$outdir/2d-$column.do") or die "cannot open $outdir/$column.do for writing";
    print G qq{
set title "Iozone performance: $column"
#set terminal png small picsize 450 350
set terminal png medium size $size2d nocrop
set logscale x
set xlabel "File size in KBytes"
set ylabel "Kbytes/sec"
set output "2d-$column.png"
};

    print HTML qq{
      <tr>
       <td align="center">
         <h2><a name="s2d-$column"></a>2d-$column</h2><a href="#top">[top]</a><BR/>
         <img src="2d-$column.png" alt="2d-$column"/><BR/>
       </td>
      </tr>
    };



    print G "plot ". join(", ", map{qq{"2d-$_" using 1:$columns{$column} title "$_" with lines}}(@datafiles));

    print G "\n";

    close G;

    if ( system("cd $outdir && gnuplot $column.do && gnuplot 2d-$column.do") )
    {
        print STDERR "(failed) ";
    }
    else
    {
        print STDERR "(ok) ";
    }
}

print HTML qq{
</table>
</body>
</html>
};
print STDERR "done.\n";
