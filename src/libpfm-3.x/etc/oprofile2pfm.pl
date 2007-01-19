#!/usr/bin/perl -w

# Written by Philip Mucci (mucci@cs.utk.edu)
# Intended to be run above the events directory of the oprofile CVS tree
# Output goes to stdout and can be used as a header file for libpfm.
# 
# Example: ./oprofile2pfm mips/*/events > gen_mips64_events.h
#

foreach $file (@ARGV)
{
    # Open this file.
    open(FILE, $file) || die "Can't open $file: $!\n";
    $proc = $file;
    $proc =~ s/(.*)\/.*/$1/g;
#    print $proc;
    LINE: while ( <FILE> )
    {
	chomp;
	if (/^\s*\#/)
	{
	    next;
	}
	elsif (/^\s*event\:.*/)
	{
	    @centry = split(":");
	    @sentry = split(" ");
	    $name = ($centry[5]);
	    $name =~ s/\s+//;
	    $desc = ($centry[6]);
	    $desc =~ s/^\s*(.*)\s*$/$1/g;
	    $tmpeven = $sentry[0];
	    $tmpcntr = $sentry[1];
#	    printf "|%s|%s|%s|%s|\n",$name,$desc,$tmpeven,$tmpcntr;
	    @even = split(":",$tmpeven);
	    @cntr = split(":",$tmpcntr);
#	    printf "0x%02x,%s\n",$even[1],$cntr[1];
	    @cntrarray = split(",",$cntr[1]);
	    if ($even[1] =~ /^0x/)
	    {
		$tmp = 0;
#		printf("Hex detected %s\n",$even[1]);
		$tmp = hex($even[1]);
#		printf("Hex converted to %d\n",$tmp);
		$even[1] = $tmp;
	    }
	    foreach $c (@cntrarray)
	    {
		$pfmeventd{$name} = $desc;
		$pfmeventc{$name} |= 1 << $c;
		$pfmevente{$name} |= $even[1] << (8*$c);
#		printf "event 0x%08x counter %d,0x%08x\n",$even[1],$c,$even[1] << (8*$c);
	    }
	}
    }

    printf "static pme_gen_mips64_entry_t gen_mips64_%s_pe\[\] = {\n",$proc;
    
    foreach $key (keys %pfmeventd)
    {
	printf "\t{.pme_name=\"%s\",\n",$key;
	printf "\t .pme_entry_code.pme_vcode = 0x%08x,\n",$pfmevente{$key};
	printf "\t .pme_counters = 0x%x,\n",$pfmeventc{$key};
	printf "\t .pme_desc = \"%s\"\n\t},\n",$pfmeventd{$key};
    }
    %pfmeventc = ();
    %pfmeventd = ();
    %pfmevente = ();

    print "};\n\n";

    close(FILE);
}
