// see Alpha Architecture Handbook for a description
	.set noat
	.set noreorder
	.text
	.align 4
	.file 1 "read_cycle_counter.s"
//
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
//
	.globl  read_virt_cycle_counter
	.ent 	read_virt_cycle_counter
read_virt_cycle_counter:
	.frame  $sp, 0, $26
	.save_ra $0
	.prologue 0
	rpcc $0
	sll $0,32,$1
	addq $0,$1,$0
	sra $0,32,$0		//   return (cc + (cc<<32)) >> 32
	ret $31,($26),1
	.end 	read_virt_cycle_counter
	.ident "$Id$"

