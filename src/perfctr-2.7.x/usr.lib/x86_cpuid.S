	.file	"x86_cpuid.S"

///	struct cpuid { unsigned int eax, ebx, edx, ecx; };
///	void get_cpuid(unsigned int eax, struct cpuid*);
///	move eax to %eax, then execute CPUID
///	copy resulting %eax, %ebx, %edx, %ecx into the cpuid buf
	.text
	.align	4
	.globl	get_cpuid
get_cpuid:
	pushl	%ebp
	pushl	%ebx
	movl	12(%esp),%eax		/* argument for CPUID */
	.byte	0x0F,0xA2		/* opcode for CPUID */
	movl	16(%esp),%ebp		/* ptr to struct cpuid buf */
	movl	%eax,(%ebp)
	movl	%ebx,4(%ebp)
	movl	%edx,8(%ebp)
	movl	%ecx,12(%ebp)
	popl	%ebx
	popl	%ebp
	ret
	.align	4
	.type	get_cpuid,@function
	.size	get_cpuid,.-get_cpuid
