	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB0:
	.text
LHOTB0:
	.align 4,0x90
	.globl __Z4loadDv4_f
__Z4loadDv4_f:
LFB0:
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
	.globl __Z4loadPKf
__Z4loadPKf:
LFB1:
	vmovups	(%rdi), %xmm0
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
	.globl __Z5storePfDv4_f
__Z5storePfDv4_f:
LFB2:
	vmovups	%xmm0, (%rdi)
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
	.globl __Z3addPfDv4_f
__Z3addPfDv4_f:
LFB3:
	vaddps	(%rdi), %xmm0, %xmm0
	vmovups	%xmm0, (%rdi)
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
	.globl __Z5load3PKf
__Z5load3PKf:
LFB4:
	vmovups	(%rdi), %xmm0
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
	.globl __Z6store3PfDv4_f
__Z6store3PfDv4_f:
LFB5:
	vblendps	$8, (%rdi), %xmm0, %xmm0
	vmovups	%xmm0, (%rdi)
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
	.globl __Z4add3PfDv4_f
__Z4add3PfDv4_f:
LFB6:
	vaddps	(%rdi), %xmm0, %xmm0
	vblendps	$8, (%rdi), %xmm0, %xmm0
	vmovups	%xmm0, (%rdi)
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
	.globl __Z5add11PfS_Dv4_f
__Z5add11PfS_Dv4_f:
LFB7:
	vaddps	(%rdi), %xmm0, %xmm1
	vmovaps	%xmm1, (%rdi)
	vaddps	(%rsi), %xmm0, %xmm0
	vaddps	%xmm1, %xmm0, %xmm0
	vmovaps	%xmm0, (%rsi)
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
	.globl __Z5add14PfS_Dv4_f
__Z5add14PfS_Dv4_f:
LFB8:
	vaddps	(%rdi), %xmm0, %xmm1
	vmovups	%xmm1, (%rdi)
	vaddps	(%rsi), %xmm0, %xmm0
	vaddps	%xmm1, %xmm0, %xmm0
	vmovups	%xmm0, (%rsi)
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
	.globl __Z5add98PfS_Dv4_f
__Z5add98PfS_Dv4_f:
LFB9:
	vaddps	(%rdi), %xmm0, %xmm1
	vmovups	%xmm1, (%rdi)
	vaddps	(%rsi), %xmm0, %xmm0
	vaddps	%xmm1, %xmm0, %xmm0
	vmovups	%xmm0, (%rsi)
	ret
LFE9:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE9:
	.text
LHOTE9:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB12:
	.text
LHOTB12:
	.align 4,0x90
	.globl __Z4doitv
__Z4doitv:
LFB10:
	pushq	%rbx
LCFI0:
	movl	$12300, %edx
	xorl	%esi, %esi
	subq	$12304, %rsp
LCFI1:
	movq	%rsp, %rdi
	call	_memset
	movl	$0x3f800000, (%rsp)
	movq	%rsp, %rdi
	movq	%rsp, %rax
	vmovaps	LC11(%rip), %xmm0
	leaq	4104(%rsp), %rdx
	.align 4,0x90
L12:
	vmovups	%xmm0, (%rax)
	addq	$12, %rax
	cmpq	%rdx, %rax
	jne	L12
	.align 4,0x90
L14:
	vaddps	(%rdi), %xmm0, %xmm1
	addq	$12, %rdi
	vmovups	%xmm1, -12(%rdi)
	cmpq	%rdx, %rdi
	jne	L14
	vcvttss2si	496(%rsp), %eax
	addq	$12304, %rsp
LCFI2:
	popq	%rbx
LCFI3:
	ret
LFE10:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE12:
	.text
LHOTE12:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB13:
	.text
LHOTB13:
	.align 4,0x90
	.globl __Z5doit3v
__Z5doit3v:
LFB11:
	pushq	%rbx
LCFI4:
	movl	$12300, %edx
	xorl	%esi, %esi
	subq	$12304, %rsp
LCFI5:
	movq	%rsp, %rdi
	call	_memset
	movl	$0x3f800000, (%rsp)
	movq	%rsp, %rdi
	movq	%rsp, %rax
	vmovaps	LC11(%rip), %xmm1
	leaq	4104(%rsp), %rdx
	.align 4,0x90
L20:
	vblendps	$8, (%rax), %xmm1, %xmm0
	addq	$12, %rax
	vmovups	%xmm0, -12(%rax)
	cmpq	%rdx, %rax
	jne	L20
	.align 4,0x90
L22:
	vaddps	(%rdi), %xmm1, %xmm0
	addq	$12, %rdi
	vblendps	$8, -12(%rdi), %xmm0, %xmm0
	vmovups	%xmm0, -12(%rdi)
	cmpq	%rdi, %rdx
	jne	L22
	vcvttss2si	496(%rsp), %eax
	addq	$12304, %rsp
LCFI6:
	popq	%rbx
LCFI7:
	ret
LFE11:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE13:
	.text
LHOTE13:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB14:
	.section __TEXT,__text_startup,regular,pure_instructions
LHOTB14:
	.align 4
	.globl _main
_main:
LFB12:
	pushq	%rbx
LCFI8:
	call	__Z4doitv
	movl	%eax, %ebx
	call	__Z5doit3v
	addl	%ebx, %eax
	popq	%rbx
LCFI9:
	ret
LFE12:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE14:
	.section __TEXT,__text_startup,regular,pure_instructions
LHOTE14:
	.literal16
	.align 4
LC11:
	.long	1065353216
	.long	1073741824
	.long	1077936128
	.long	1082130432
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
	.align 3
LEFDE9:
LSFDE11:
	.set L$set$11,LEFDE11-LASFDE11
	.long L$set$11
LASFDE11:
	.long	LASFDE11-EH_frame1
	.quad	LFB5-.
	.set L$set$12,LFE5-LFB5
	.quad L$set$12
	.byte	0
	.align 3
LEFDE11:
LSFDE13:
	.set L$set$13,LEFDE13-LASFDE13
	.long L$set$13
LASFDE13:
	.long	LASFDE13-EH_frame1
	.quad	LFB6-.
	.set L$set$14,LFE6-LFB6
	.quad L$set$14
	.byte	0
	.align 3
LEFDE13:
LSFDE15:
	.set L$set$15,LEFDE15-LASFDE15
	.long L$set$15
LASFDE15:
	.long	LASFDE15-EH_frame1
	.quad	LFB7-.
	.set L$set$16,LFE7-LFB7
	.quad L$set$16
	.byte	0
	.align 3
LEFDE15:
LSFDE17:
	.set L$set$17,LEFDE17-LASFDE17
	.long L$set$17
LASFDE17:
	.long	LASFDE17-EH_frame1
	.quad	LFB8-.
	.set L$set$18,LFE8-LFB8
	.quad L$set$18
	.byte	0
	.align 3
LEFDE17:
LSFDE19:
	.set L$set$19,LEFDE19-LASFDE19
	.long L$set$19
LASFDE19:
	.long	LASFDE19-EH_frame1
	.quad	LFB9-.
	.set L$set$20,LFE9-LFB9
	.quad L$set$20
	.byte	0
	.align 3
LEFDE19:
LSFDE21:
	.set L$set$21,LEFDE21-LASFDE21
	.long L$set$21
LASFDE21:
	.long	LASFDE21-EH_frame1
	.quad	LFB10-.
	.set L$set$22,LFE10-LFB10
	.quad L$set$22
	.byte	0
	.byte	0x4
	.set L$set$23,LCFI0-LFB10
	.long L$set$23
	.byte	0xe
	.byte	0x10
	.byte	0x83
	.byte	0x2
	.byte	0x4
	.set L$set$24,LCFI1-LCFI0
	.long L$set$24
	.byte	0xe
	.byte	0xa0,0x60
	.byte	0x4
	.set L$set$25,LCFI2-LCFI1
	.long L$set$25
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$26,LCFI3-LCFI2
	.long L$set$26
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE21:
LSFDE23:
	.set L$set$27,LEFDE23-LASFDE23
	.long L$set$27
LASFDE23:
	.long	LASFDE23-EH_frame1
	.quad	LFB11-.
	.set L$set$28,LFE11-LFB11
	.quad L$set$28
	.byte	0
	.byte	0x4
	.set L$set$29,LCFI4-LFB11
	.long L$set$29
	.byte	0xe
	.byte	0x10
	.byte	0x83
	.byte	0x2
	.byte	0x4
	.set L$set$30,LCFI5-LCFI4
	.long L$set$30
	.byte	0xe
	.byte	0xa0,0x60
	.byte	0x4
	.set L$set$31,LCFI6-LCFI5
	.long L$set$31
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$32,LCFI7-LCFI6
	.long L$set$32
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE23:
LSFDE25:
	.set L$set$33,LEFDE25-LASFDE25
	.long L$set$33
LASFDE25:
	.long	LASFDE25-EH_frame1
	.quad	LFB12-.
	.set L$set$34,LFE12-LFB12
	.quad L$set$34
	.byte	0
	.byte	0x4
	.set L$set$35,LCFI8-LFB12
	.long L$set$35
	.byte	0xe
	.byte	0x10
	.byte	0x83
	.byte	0x2
	.byte	0x4
	.set L$set$36,LCFI9-LCFI8
	.long L$set$36
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE25:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
