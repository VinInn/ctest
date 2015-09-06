	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB0:
	.text
LHOTB0:
	.align 4,0x90
	.globl __Z3fmafff
__Z3fmafff:
LFB0:
	mulss	%xmm1, %xmm2
	addss	%xmm2, %xmm0
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
	.globl __Z3fmafff.arch_haswell
__Z3fmafff.arch_haswell:
LFB1:
	vfmadd231ss	%xmm2, %xmm1, %xmm0
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
	.globl __Z3fmafff.arch_bdver1
__Z3fmafff.arch_bdver1:
LFB2:
	vfmaddss	%xmm0, %xmm2, %xmm1, %xmm0
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
	.globl __ZGVbN4vvv__Z3fmafff.arch_bdver1
__ZGVbN4vvv__Z3fmafff.arch_bdver1:
LFB3:
	vfmaddps	%xmm0, %xmm2, %xmm1, %xmm0
	ret
LFE3:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE3:
	.text
LHOTE3:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB4:
	.text
LHOTB4:
	.align 4,0x90
	.globl __ZGVcN8vvv__Z3fmafff.arch_bdver1
__ZGVcN8vvv__Z3fmafff.arch_bdver1:
LFB4:
	leaq	8(%rsp), %r10
LCFI0:
	andq	$-32, %rsp
	vfmaddps	%ymm0, %ymm2, %ymm1, %ymm0
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
LFE4:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE4:
	.text
LHOTE4:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB5:
	.text
LHOTB5:
	.align 4,0x90
	.globl __ZGVdN8vvv__Z3fmafff.arch_bdver1
__ZGVdN8vvv__Z3fmafff.arch_bdver1:
LFB5:
	leaq	8(%rsp), %r10
LCFI5:
	andq	$-32, %rsp
	vmulps	%ymm2, %ymm1, %ymm1
	pushq	-8(%r10)
	pushq	%rbp
	vaddps	%ymm0, %ymm1, %ymm0
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
LFE5:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE5:
	.text
LHOTE5:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB6:
	.text
LHOTB6:
	.align 4,0x90
	.globl __ZGVbN4vvv__Z3fmafff.arch_haswell
__ZGVbN4vvv__Z3fmafff.arch_haswell:
LFB6:
	vfmadd231ps	%xmm2, %xmm1, %xmm0
	ret
LFE6:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE6:
	.text
LHOTE6:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB7:
	.text
LHOTB7:
	.align 4,0x90
	.globl __ZGVcN8vvv__Z3fmafff.arch_haswell
__ZGVcN8vvv__Z3fmafff.arch_haswell:
LFB7:
	leaq	8(%rsp), %r10
LCFI10:
	andq	$-32, %rsp
	vfmadd231ps	%ymm2, %ymm1, %ymm0
	pushq	-8(%r10)
	pushq	%rbp
LCFI11:
	movq	%rsp, %rbp
	pushq	%r10
LCFI12:
	popq	%r10
LCFI13:
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI14:
	ret
LFE7:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE7:
	.text
LHOTE7:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB8:
	.text
LHOTB8:
	.align 4,0x90
	.globl __ZGVdN8vvv__Z3fmafff.arch_haswell
__ZGVdN8vvv__Z3fmafff.arch_haswell:
LFB8:
	leaq	8(%rsp), %r10
LCFI15:
	andq	$-32, %rsp
	vfmadd231ps	%ymm2, %ymm1, %ymm0
	pushq	-8(%r10)
	pushq	%rbp
LCFI16:
	movq	%rsp, %rbp
	pushq	%r10
LCFI17:
	popq	%r10
LCFI18:
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI19:
	ret
LFE8:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE8:
	.text
LHOTE8:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB9:
	.text
LHOTB9:
	.align 4,0x90
	.globl __ZGVbN4vvv__Z3fmafff
__ZGVbN4vvv__Z3fmafff:
LFB9:
	mulps	%xmm1, %xmm2
	addps	%xmm2, %xmm0
	ret
LFE9:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE9:
	.text
LHOTE9:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB10:
	.text
LHOTB10:
	.align 4,0x90
	.globl __ZGVcN8vvv__Z3fmafff
__ZGVcN8vvv__Z3fmafff:
LFB10:
	leaq	8(%rsp), %r10
LCFI20:
	andq	$-32, %rsp
	vmulps	%ymm2, %ymm1, %ymm1
	pushq	-8(%r10)
	pushq	%rbp
	vaddps	%ymm0, %ymm1, %ymm0
LCFI21:
	movq	%rsp, %rbp
	pushq	%r10
LCFI22:
	popq	%r10
LCFI23:
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI24:
	ret
LFE10:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE10:
	.text
LHOTE10:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB11:
	.text
LHOTB11:
	.align 4,0x90
	.globl __ZGVdN8vvv__Z3fmafff
__ZGVdN8vvv__Z3fmafff:
LFB11:
	leaq	8(%rsp), %r10
LCFI25:
	andq	$-32, %rsp
	vmulps	%ymm2, %ymm1, %ymm1
	pushq	-8(%r10)
	pushq	%rbp
	vaddps	%ymm0, %ymm1, %ymm0
LCFI26:
	movq	%rsp, %rbp
	pushq	%r10
LCFI27:
	popq	%r10
LCFI28:
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI29:
	ret
LFE11:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE11:
	.text
LHOTE11:
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
LSFDE9:
	.set L$set$9,LEFDE9-LASFDE9
	.long L$set$9
LASFDE9:
	.long	LASFDE9-EH_frame1
	.quad	LFB4-.
	.set L$set$10,LFE4-LFB4
	.quad L$set$10
	.byte	0
	.byte	0x4
	.set L$set$11,LCFI0-LFB4
	.long L$set$11
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$12,LCFI1-LCFI0
	.long L$set$12
	.byte	0x10
	.byte	0x6
	.byte	0x2
	.byte	0x76
	.byte	0
	.byte	0x4
	.set L$set$13,LCFI2-LCFI1
	.long L$set$13
	.byte	0xf
	.byte	0x3
	.byte	0x76
	.byte	0x78
	.byte	0x6
	.byte	0x4
	.set L$set$14,LCFI3-LCFI2
	.long L$set$14
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$15,LCFI4-LCFI3
	.long L$set$15
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.align 3
LEFDE9:
LSFDE11:
	.set L$set$16,LEFDE11-LASFDE11
	.long L$set$16
LASFDE11:
	.long	LASFDE11-EH_frame1
	.quad	LFB5-.
	.set L$set$17,LFE5-LFB5
	.quad L$set$17
	.byte	0
	.byte	0x4
	.set L$set$18,LCFI5-LFB5
	.long L$set$18
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$19,LCFI6-LCFI5
	.long L$set$19
	.byte	0x10
	.byte	0x6
	.byte	0x2
	.byte	0x76
	.byte	0
	.byte	0x4
	.set L$set$20,LCFI7-LCFI6
	.long L$set$20
	.byte	0xf
	.byte	0x3
	.byte	0x76
	.byte	0x78
	.byte	0x6
	.byte	0x4
	.set L$set$21,LCFI8-LCFI7
	.long L$set$21
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$22,LCFI9-LCFI8
	.long L$set$22
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.align 3
LEFDE11:
LSFDE13:
	.set L$set$23,LEFDE13-LASFDE13
	.long L$set$23
LASFDE13:
	.long	LASFDE13-EH_frame1
	.quad	LFB6-.
	.set L$set$24,LFE6-LFB6
	.quad L$set$24
	.byte	0
	.align 3
LEFDE13:
LSFDE15:
	.set L$set$25,LEFDE15-LASFDE15
	.long L$set$25
LASFDE15:
	.long	LASFDE15-EH_frame1
	.quad	LFB7-.
	.set L$set$26,LFE7-LFB7
	.quad L$set$26
	.byte	0
	.byte	0x4
	.set L$set$27,LCFI10-LFB7
	.long L$set$27
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$28,LCFI11-LCFI10
	.long L$set$28
	.byte	0x10
	.byte	0x6
	.byte	0x2
	.byte	0x76
	.byte	0
	.byte	0x4
	.set L$set$29,LCFI12-LCFI11
	.long L$set$29
	.byte	0xf
	.byte	0x3
	.byte	0x76
	.byte	0x78
	.byte	0x6
	.byte	0x4
	.set L$set$30,LCFI13-LCFI12
	.long L$set$30
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$31,LCFI14-LCFI13
	.long L$set$31
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.align 3
LEFDE15:
LSFDE17:
	.set L$set$32,LEFDE17-LASFDE17
	.long L$set$32
LASFDE17:
	.long	LASFDE17-EH_frame1
	.quad	LFB8-.
	.set L$set$33,LFE8-LFB8
	.quad L$set$33
	.byte	0
	.byte	0x4
	.set L$set$34,LCFI15-LFB8
	.long L$set$34
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$35,LCFI16-LCFI15
	.long L$set$35
	.byte	0x10
	.byte	0x6
	.byte	0x2
	.byte	0x76
	.byte	0
	.byte	0x4
	.set L$set$36,LCFI17-LCFI16
	.long L$set$36
	.byte	0xf
	.byte	0x3
	.byte	0x76
	.byte	0x78
	.byte	0x6
	.byte	0x4
	.set L$set$37,LCFI18-LCFI17
	.long L$set$37
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$38,LCFI19-LCFI18
	.long L$set$38
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.align 3
LEFDE17:
LSFDE19:
	.set L$set$39,LEFDE19-LASFDE19
	.long L$set$39
LASFDE19:
	.long	LASFDE19-EH_frame1
	.quad	LFB9-.
	.set L$set$40,LFE9-LFB9
	.quad L$set$40
	.byte	0
	.align 3
LEFDE19:
LSFDE21:
	.set L$set$41,LEFDE21-LASFDE21
	.long L$set$41
LASFDE21:
	.long	LASFDE21-EH_frame1
	.quad	LFB10-.
	.set L$set$42,LFE10-LFB10
	.quad L$set$42
	.byte	0
	.byte	0x4
	.set L$set$43,LCFI20-LFB10
	.long L$set$43
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$44,LCFI21-LCFI20
	.long L$set$44
	.byte	0x10
	.byte	0x6
	.byte	0x2
	.byte	0x76
	.byte	0
	.byte	0x4
	.set L$set$45,LCFI22-LCFI21
	.long L$set$45
	.byte	0xf
	.byte	0x3
	.byte	0x76
	.byte	0x78
	.byte	0x6
	.byte	0x4
	.set L$set$46,LCFI23-LCFI22
	.long L$set$46
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$47,LCFI24-LCFI23
	.long L$set$47
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.align 3
LEFDE21:
LSFDE23:
	.set L$set$48,LEFDE23-LASFDE23
	.long L$set$48
LASFDE23:
	.long	LASFDE23-EH_frame1
	.quad	LFB11-.
	.set L$set$49,LFE11-LFB11
	.quad L$set$49
	.byte	0
	.byte	0x4
	.set L$set$50,LCFI25-LFB11
	.long L$set$50
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$51,LCFI26-LCFI25
	.long L$set$51
	.byte	0x10
	.byte	0x6
	.byte	0x2
	.byte	0x76
	.byte	0
	.byte	0x4
	.set L$set$52,LCFI27-LCFI26
	.long L$set$52
	.byte	0xf
	.byte	0x3
	.byte	0x76
	.byte	0x78
	.byte	0x6
	.byte	0x4
	.set L$set$53,LCFI28-LCFI27
	.long L$set$53
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$54,LCFI29-LCFI28
	.long L$set$54
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.align 3
LEFDE23:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
