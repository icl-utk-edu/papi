	.set noat
	.set noreorder
	.text
	.align 4
	.file 1 "cpu_version.s"
	.globl  cpu_implementation_version
	.ent 	cpu_implementation_version
cpu_implementation_version:
	.frame  $sp, 0, $26
	.prologue 0
	implver $0	
	ret	($26)
	.end 	cpu_implementation_version


