#!/usr/bin/perl

sub trim($)
{
	my $string = shift;
	$string =~ s/^\s+//;
	$string =~ s/\s+$//;
	return $string;
}

sub remove_brackets_and_paranthesis ($)
{
	my $string = shift;
	$string =~ s/\(.*\)//g;
	$string =~ s/\[.*\]//g;
	return $string;
}

sub change_dot_per_underscore ($)
{
	my $string = shift;
	$string =~ s/\./\_/g;
	return $string;
}

if (@ARGV!=2)
{
	die "You must pass a file containing the section of FreeBSD pmc manual referring to Event specifiers and the architecture (ATOM/CORE2/..) type"
}
else
{
	$file = $ARGV[0];
	$type = $ARGV[1];
}

open FILE, $file;
open FILE_C, ">new.c";
open FILE_H, ">new.h";
print FILE_C "{\n";
while ($counter = <FILE>)
{
	$counter = trim(remove_brackets_and_paranthesis($counter));
	print "EVENT NAME: ".$counter;
	$description = <FILE>;
	$description = trim(remove_brackets_and_paranthesis($description));
	print " DESCRIPTION: ".$description."\n";
	$foo = <FILE>;

	print FILE_C "\t{\"".$counter."\", \"".$description."\"},\n";
	print FILE_H "\tPNE_".$type."_".uc(change_dot_per_underscore($counter)).",\n";
}
print FILE_C "\t{ NULL, NULL }\n};";
print FILE_H "\tPNE_".$type."_NATNAME_GUARD\n};";
close FILE;
close FILE_C;
close FILE_H;

print "new.c and new.h have been created\n";

exit;
