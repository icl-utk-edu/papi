// see Alpha Architecture Handbook for a description
	.set noat
	.set noreorder
	.text
	.align 4
	.file 1 "read_cycle_counter.s"
	.globl  read_cycle_counter
	.ent 	read_cycle_counter
read_cycle_counter:
	.frame  $sp, 0, $26
	.save_ra $0
	.prologue 0
	rpcc	$0		//  read processor cycle counter
	zapnot	$0,15,$0	//  mask out upper 32 bits
	ret	($26)
	.end 	read_cycle_counter


