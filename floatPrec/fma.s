	.text
	.align 4,0x90
	.globl __Z3foofff
__Z3foofff:
LFB101:
	vfmadd132ss	%xmm1, %xmm2, %xmm0
	ret
LFE101:
	.align 4,0x90
	.globl __Z3barfff
__Z3barfff:
LFB102:
	vfmadd132ss	%xmm1, %xmm2, %xmm0
	ret
LFE102:
	.section __TEXT,__textcoal_nt,coalesced,pure_instructions
	.align 4
	.globl __Z5nofmafff
	.weak_definition __Z5nofmafff
__Z5nofmafff:
LFB103:
	vmulss	%xmm1, %xmm0, %xmm1
	vaddss	%xmm2, %xmm1, %xmm0
	ret
LFE103:
	.text
	.align 4,0x90
	.globl __Z4polyPKffff
__Z4polyPKffff:
LFB104:
	vmovaps	%xmm0, %xmm3
	vmovaps	%xmm1, %xmm0
	vmovss	8(%rdi), %xmm1
	vfmadd213ss	4(%rdi), %xmm3, %xmm1
	vfmadd213ss	(%rdi), %xmm3, %xmm1
	jmp	__Z5nofmafff
LFE104:
	.align 4,0x90
	.globl __Z4loopPKfff
__Z4loopPKfff:
LFB105:
	pushq	%r15
LCFI0:
	vmovd	%xmm1, %r15d
	pushq	%r14
LCFI1:
	vmovd	%xmm0, %r14d
	pushq	%r13
LCFI2:
	leaq	_xx(%rip), %r13
	pushq	%r12
LCFI3:
	leaq	_yy(%rip), %r12
	pushq	%rbp
LCFI4:
	movq	%rdi, %rbp
	pushq	%rbx
LCFI5:
	xorl	%ebx, %ebx
	subq	$8, %rsp
LCFI6:
	.align 4,0x90
L6:
	vmovss	0(%r13,%rbx), %xmm0
	vmovd	%r15d, %xmm2
	vmovss	8(%rbp), %xmm1
	vfmadd213ss	4(%rbp), %xmm0, %xmm1
	vfmadd213ss	0(%rbp), %xmm0, %xmm1
	vmovd	%r14d, %xmm0
	call	__Z5nofmafff
	vmovss	%xmm0, (%r12,%rbx)
	addq	$4, %rbx
	cmpq	$4096, %rbx
	jne	L6
	addq	$8, %rsp
LCFI7:
	popq	%rbx
LCFI8:
	popq	%rbp
LCFI9:
	popq	%r12
LCFI10:
	popq	%r13
LCFI11:
	popq	%r14
LCFI12:
	popq	%r15
LCFI13:
	ret
LFE105:
	.globl _yy
	.zerofill __DATA,__pu_bss5,_yy,4096,5
	.globl _xx
	.zerofill __DATA,__pu_bss5,_xx,4096,5
	.section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
EH_frame1:
	.set L$set$0,LECIE1-LSCIE1
	.long L$set$0
LSCIE1:
	.long	0
	.byte	0x1
	.ascii "zR\0"
	.byte	0x1
	.byte	0x78
	.byte	0x10
	.byte	0x1
	.byte	0x10
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.byte	0x90
	.byte	0x1
	.align 3
LECIE1:
LSFDE1:
	.set L$set$1,LEFDE1-LASFDE1
	.long L$set$1
LASFDE1:
	.long	LASFDE1-EH_frame1
	.quad	LFB101-.
	.set L$set$2,LFE101-LFB101
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB102-.
	.set L$set$4,LFE102-LFB102
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB103-.
	.set L$set$6,LFE103-LFB103
	.quad L$set$6
	.byte	0
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$7,LEFDE7-LASFDE7
	.long L$set$7
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB104-.
	.set L$set$8,LFE104-LFB104
	.quad L$set$8
	.byte	0
	.align 3
LEFDE7:
LSFDE9:
	.set L$set$9,LEFDE9-LASFDE9
	.long L$set$9
LASFDE9:
	.long	LASFDE9-EH_frame1
	.quad	LFB105-.
	.set L$set$10,LFE105-LFB105
	.quad L$set$10
	.byte	0
	.byte	0x4
	.set L$set$11,LCFI0-LFB105
	.long L$set$11
	.byte	0xe
	.byte	0x10
	.byte	0x8f
	.byte	0x2
	.byte	0x4
	.set L$set$12,LCFI1-LCFI0
	.long L$set$12
	.byte	0xe
	.byte	0x18
	.byte	0x8e
	.byte	0x3
	.byte	0x4
	.set L$set$13,LCFI2-LCFI1
	.long L$set$13
	.byte	0xe
	.byte	0x20
	.byte	0x8d
	.byte	0x4
	.byte	0x4
	.set L$set$14,LCFI3-LCFI2
	.long L$set$14
	.byte	0xe
	.byte	0x28
	.byte	0x8c
	.byte	0x5
	.byte	0x4
	.set L$set$15,LCFI4-LCFI3
	.long L$set$15
	.byte	0xe
	.byte	0x30
	.byte	0x86
	.byte	0x6
	.byte	0x4
	.set L$set$16,LCFI5-LCFI4
	.long L$set$16
	.byte	0xe
	.byte	0x38
	.byte	0x83
	.byte	0x7
	.byte	0x4
	.set L$set$17,LCFI6-LCFI5
	.long L$set$17
	.byte	0xe
	.byte	0x40
	.byte	0x4
	.set L$set$18,LCFI7-LCFI6
	.long L$set$18
	.byte	0xe
	.byte	0x38
	.byte	0x4
	.set L$set$19,LCFI8-LCFI7
	.long L$set$19
	.byte	0xe
	.byte	0x30
	.byte	0x4
	.set L$set$20,LCFI9-LCFI8
	.long L$set$20
	.byte	0xe
	.byte	0x28
	.byte	0x4
	.set L$set$21,LCFI10-LCFI9
	.long L$set$21
	.byte	0xe
	.byte	0x20
	.byte	0x4
	.set L$set$22,LCFI11-LCFI10
	.long L$set$22
	.byte	0xe
	.byte	0x18
	.byte	0x4
	.set L$set$23,LCFI12-LCFI11
	.long L$set$23
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$24,LCFI13-LCFI12
	.long L$set$24
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE9:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
