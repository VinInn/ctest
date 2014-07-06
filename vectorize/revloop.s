	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB1:
	.text
LHOTB1:
	.align 4,0x90
	.globl __Z3foov
__Z3foov:
LFB0:
	vmovdqa	LC0(%rip), %ymm2
	xorl	%eax, %eax
	leaq	4064+_x(%rip), %rdx
	leaq	4064+_y(%rip), %rsi
	leaq	2020+_z(%rip), %rcx
	.align 4,0x90
L2:
	vpermd	(%rdx,%rax), %ymm2, %ymm0
	vpermd	(%rcx,%rax), %ymm2, %ymm1
	vpermd	(%rsi,%rax), %ymm2, %ymm3
	vfmadd231ps	%ymm1, %ymm3, %ymm0
	vpermd	%ymm0, %ymm2, %ymm0
	vmovaps	%ymm0, (%rdx,%rax)
	subq	$32, %rax
	cmpq	$-2048, %rax
	jne	L2
	vzeroupper
	ret
LFE0:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE1:
	.text
LHOTE1:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB2:
	.text
LHOTB2:
	.align 4,0x90
	.globl __Z4foo2v
__Z4foo2v:
LFB1:
	leaq	2048+_x(%rip), %rdx
	xorl	%eax, %eax
	leaq	4+_z(%rip), %rsi
	leaq	2048+_y(%rip), %rcx
	.align 4,0x90
L6:
	vmovaps	(%rdx,%rax), %ymm1
	vmovups	(%rsi,%rax), %ymm0
	vfmadd132ps	(%rcx,%rax), %ymm1, %ymm0
	vmovaps	%ymm0, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$2048, %rax
	jne	L6
	vzeroupper
	ret
LFE1:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE2:
	.text
LHOTE2:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB3:
	.text
LHOTB3:
	.align 4,0x90
	.globl __Z3barv
__Z3barv:
LFB2:
	vmovdqa	LC0(%rip), %ymm1
	leaq	2048+_z(%rip), %rdx
	leaq	_y(%rip), %rcx
	leaq	4064+_x(%rip), %rax
	leaq	4096+_z(%rip), %rsi
	.align 4,0x90
L9:
	vmovaps	(%rdx), %ymm2
	addq	$32, %rdx
	vpermd	(%rax), %ymm1, %ymm0
	addq	$32, %rcx
	vfmadd231ps	-32(%rcx), %ymm2, %ymm0
	subq	$32, %rax
	vpermd	%ymm0, %ymm1, %ymm0
	vmovaps	%ymm0, 32(%rax)
	cmpq	%rsi, %rdx
	jne	L9
	vzeroupper
	ret
LFE2:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE3:
	.text
LHOTE3:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB4:
	.text
LHOTB4:
	.align 4,0x90
	.globl __Z4bar2v
__Z4bar2v:
LFB3:
	vmovdqa	LC0(%rip), %ymm1
	leaq	4064+_z(%rip), %rdx
	leaq	2016+_y(%rip), %rcx
	leaq	2048+_x(%rip), %rax
	leaq	2016+_z(%rip), %rsi
	.align 4,0x90
L12:
	vpermd	(%rdx), %ymm1, %ymm0
	subq	$32, %rdx
	vpermd	(%rcx), %ymm1, %ymm2
	addq	$32, %rax
	vfmadd213ps	-32(%rax), %ymm2, %ymm0
	subq	$32, %rcx
	vmovaps	%ymm0, -32(%rax)
	cmpq	%rsi, %rdx
	jne	L12
	vzeroupper
	ret
LFE3:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE4:
	.text
LHOTE4:
	.globl _z
	.zerofill __DATA,__pu_bss6,_z,4096,6
	.globl _y
	.zerofill __DATA,__pu_bss6,_y,4096,6
	.globl _x
	.zerofill __DATA,__pu_bss6,_x,4096,6
	.const
	.align 5
LC0:
	.long	7
	.long	6
	.long	5
	.long	4
	.long	3
	.long	2
	.long	1
	.long	0
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
	.quad	LFB0-.
	.set L$set$2,LFE0-LFB0
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB1-.
	.set L$set$4,LFE1-LFB1
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB2-.
	.set L$set$6,LFE2-LFB2
	.quad L$set$6
	.byte	0
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$7,LEFDE7-LASFDE7
	.long L$set$7
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB3-.
	.set L$set$8,LFE3-LFB3
	.quad L$set$8
	.byte	0
	.align 3
LEFDE7:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
