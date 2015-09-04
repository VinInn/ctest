	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB0:
	.text
LHOTB0:
	.align 4,0x90
	.globl __Z3fmafff
__Z3fmafff:
LFB0:
	vfmadd231ss	%xmm2, %xmm1, %xmm0
	ret
LFE0:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE0:
	.text
LHOTE0:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB1:
	.text
LHOTB1:
	.align 4,0x90
	.globl __ZGVbN4vvv__Z3fmafff
__ZGVbN4vvv__Z3fmafff:
LFB1:
	vfmadd231ps	%xmm2, %xmm1, %xmm0
	ret
LFE1:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE1:
	.text
LHOTE1:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB2:
	.text
LHOTB2:
	.align 4,0x90
	.globl __ZGVcN8vvv__Z3fmafff
__ZGVcN8vvv__Z3fmafff:
LFB2:
	leaq	8(%rsp), %r10
LCFI0:
	andq	$-32, %rsp
	vfmadd231ps	%ymm2, %ymm1, %ymm0
	pushq	-8(%r10)
	pushq	%rbp
LCFI1:
	movq	%rsp, %rbp
	pushq	%r10
LCFI2:
	popq	%r10
LCFI3:
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI4:
	ret
LFE2:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE2:
	.text
LHOTE2:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB3:
	.text
LHOTB3:
	.align 4,0x90
	.globl __ZGVdN8vvv__Z3fmafff
__ZGVdN8vvv__Z3fmafff:
LFB3:
	leaq	8(%rsp), %r10
LCFI5:
	andq	$-32, %rsp
	vfmadd231ps	%ymm2, %ymm1, %ymm0
	pushq	-8(%r10)
	pushq	%rbp
LCFI6:
	movq	%rsp, %rbp
	pushq	%r10
LCFI7:
	popq	%r10
LCFI8:
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI9:
	ret
LFE3:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE3:
	.text
LHOTE3:
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
	.byte	0x4
	.set L$set$7,LCFI0-LFB2
	.long L$set$7
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$8,LCFI1-LCFI0
	.long L$set$8
	.byte	0x10
	.byte	0x6
	.byte	0x2
	.byte	0x76
	.byte	0
	.byte	0x4
	.set L$set$9,LCFI2-LCFI1
	.long L$set$9
	.byte	0xf
	.byte	0x3
	.byte	0x76
	.byte	0x78
	.byte	0x6
	.byte	0x4
	.set L$set$10,LCFI3-LCFI2
	.long L$set$10
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$11,LCFI4-LCFI3
	.long L$set$11
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$12,LEFDE7-LASFDE7
	.long L$set$12
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB3-.
	.set L$set$13,LFE3-LFB3
	.quad L$set$13
	.byte	0
	.byte	0x4
	.set L$set$14,LCFI5-LFB3
	.long L$set$14
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$15,LCFI6-LCFI5
	.long L$set$15
	.byte	0x10
	.byte	0x6
	.byte	0x2
	.byte	0x76
	.byte	0
	.byte	0x4
	.set L$set$16,LCFI7-LCFI6
	.long L$set$16
	.byte	0xf
	.byte	0x3
	.byte	0x76
	.byte	0x78
	.byte	0x6
	.byte	0x4
	.set L$set$17,LCFI8-LCFI7
	.long L$set$17
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$18,LCFI9-LCFI8
	.long L$set$18
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.align 3
LEFDE7:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
