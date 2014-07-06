	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB12:
	.text
LHOTB12:
	.align 4,0x90
	.globl __Z7computev
__Z7computev:
LFB4:
	vmovss	LC4(%rip), %xmm8
	xorl	%eax, %eax
	vxorps	%xmm4, %xmm4, %xmm4
	leaq	_x(%rip), %rcx
	vmovss	LC3(%rip), %xmm13
	leaq	_y(%rip), %rdx
	vmovaps	%xmm8, %xmm14
	vmovss	LC5(%rip), %xmm12
	vmovss	LC6(%rip), %xmm11
	vmovss	LC7(%rip), %xmm10
	vmovss	LC8(%rip), %xmm9
	vmovss	LC9(%rip), %xmm7
	vmovss	LC10(%rip), %xmm6
	vmovss	LC0(%rip), %xmm5
	jmp	L2
	.align 4,0x90
L19:
	vmovss	LC1(%rip), %xmm3
	vmovaps	%xmm5, %xmm2
	vmovss	LC2(%rip), %xmm1
L5:
	vfmadd132ss	%xmm0, %xmm3, %xmm1
	vfmadd132ss	%xmm1, %xmm2, %xmm0
	vmovss	%xmm0, (%rdx,%rax)
	addq	$4, %rax
	cmpq	$4096, %rax
	je	L18
L2:
	vmovss	(%rcx,%rax), %xmm0
	vcomiss	%xmm4, %xmm0
	jbe	L14
	vcomiss	LC1(%rip), %xmm0
	ja	L19
	vmovaps	%xmm7, %xmm2
	vmovaps	%xmm14, %xmm3
	vmovaps	%xmm6, %xmm1
	jmp	L5
	.align 4,0x90
L14:
	vcomiss	LC7(%rip), %xmm0
	jbe	L16
	vmovaps	%xmm11, %xmm2
	vmovaps	%xmm10, %xmm3
	vmovaps	%xmm9, %xmm1
	jmp	L5
	.align 4,0x90
L16:
	vmovaps	%xmm13, %xmm2
	vmovaps	%xmm8, %xmm3
	vmovaps	%xmm12, %xmm1
	jmp	L5
	.align 4,0x90
L18:
	ret
LFE4:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE12:
	.text
LHOTE12:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB19:
	.text
LHOTB19:
	.align 4,0x90
	.globl __Z8compute2v
__Z8compute2v:
LFB5:
	leaq	8(%rsp), %r10
LCFI0:
	andq	$-32, %rsp
	xorl	%eax, %eax
	vmovaps	LC13(%rip), %ymm9
	pushq	-8(%r10)
	pushq	%rbp
	vxorps	%xmm3, %xmm3, %xmm3
	leaq	_x(%rip), %rcx
	vmovaps	LC14(%rip), %ymm8
LCFI1:
	movq	%rsp, %rbp
	vmovaps	LC15(%rip), %ymm7
	pushq	%r10
LCFI2:
	leaq	_y(%rip), %rdx
	vmovaps	LC16(%rip), %ymm6
	vmovaps	LC17(%rip), %ymm5
	vmovaps	LC18(%rip), %ymm4
	.align 4,0x90
L21:
	vmovaps	(%rcx,%rax), %ymm0
	vmovaps	%ymm0, %ymm2
	vmovaps	%ymm0, %ymm1
	vfmadd132ps	%ymm9, %ymm8, %ymm2
	vfmadd132ps	%ymm6, %ymm5, %ymm1
	vfmadd132ps	%ymm0, %ymm7, %ymm2
	vfmadd132ps	%ymm0, %ymm4, %ymm1
	vcmpltps	%ymm0, %ymm3, %ymm0
	vblendvps	%ymm0, %ymm2, %ymm1, %ymm0
	vmovaps	%ymm0, (%rdx,%rax)
	addq	$32, %rax
	cmpq	$4096, %rax
	jne	L21
	vzeroupper
	popq	%r10
LCFI3:
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI4:
	ret
LFE5:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE19:
	.text
LHOTE19:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB20:
	.text
LHOTB20:
	.align 4,0x90
	.globl __Z8compute3v
__Z8compute3v:
LFB6:
	xorl	%edx, %edx
	vxorps	%xmm2, %xmm2, %xmm2
	movq	__ZZ5poly3fE1c@GOTPCREL(%rip), %rdi
	leaq	_x(%rip), %r9
	movq	__ZZ5poly3fE1b@GOTPCREL(%rip), %rsi
	leaq	_y(%rip), %r8
	movq	__ZZ5poly3fE1a@GOTPCREL(%rip), %rcx
	jmp	L25
	.align 4,0x90
L34:
	xorl	%eax, %eax
	vcomiss	LC1(%rip), %xmm0
	setbe	%al
L28:
	cltq
	vmovss	(%rdi,%rax,4), %xmm1
	vfmadd213ss	(%rsi,%rax,4), %xmm0, %xmm1
	vfmadd213ss	(%rcx,%rax,4), %xmm1, %xmm0
	vmovss	%xmm0, (%r8,%rdx)
	addq	$4, %rdx
	cmpq	$4096, %rdx
	je	L33
L25:
	vmovss	(%r9,%rdx), %xmm0
	vcomiss	%xmm2, %xmm0
	ja	L34
	xorl	%eax, %eax
	vcomiss	LC7(%rip), %xmm0
	setbe	%al
	addl	$2, %eax
	jmp	L28
	.align 4,0x90
L33:
	ret
LFE6:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE20:
	.text
LHOTE20:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDB26:
	.text
LHOTB26:
	.align 4,0x90
	.globl __Z4polyU8__vectorf
__Z4polyU8__vectorf:
LFB7:
	vmovaps	LC14(%rip), %ymm3
	vxorps	%xmm4, %xmm4, %xmm4
	vcmpltps	%ymm0, %ymm4, %ymm4
	leaq	8(%rsp), %r10
LCFI5:
	vmovaps	LC17(%rip), %ymm2
	andq	$-32, %rsp
	pushq	-8(%r10)
	vcmpltps	%ymm0, %ymm3, %ymm7
	pushq	%rbp
	vmovaps	LC18(%rip), %ymm5
LCFI6:
	movq	%rsp, %rbp
	pushq	%r10
LCFI7:
	popq	%r10
LCFI8:
	vblendvps	%ymm7, %ymm3, %ymm2, %ymm1
	popq	%rbp
	vblendvps	%ymm7, LC15(%rip), %ymm5, %ymm5
	vmovaps	LC21(%rip), %ymm3
	vcmpltps	%ymm0, %ymm3, %ymm6
	leaq	-8(%r10), %rsp
LCFI9:
	vblendvps	%ymm6, %ymm3, %ymm2, %ymm2
	vblendvps	%ymm4, %ymm1, %ymm2, %ymm3
	vmovaps	LC16(%rip), %ymm1
	vmovaps	LC23(%rip), %ymm2
	vblendvps	%ymm7, LC13(%rip), %ymm1, %ymm1
	vblendvps	%ymm6, LC22(%rip), %ymm2, %ymm2
	vblendvps	%ymm4, %ymm1, %ymm2, %ymm1
	vfmadd132ps	%ymm0, %ymm3, %ymm1
	vmovaps	LC25(%rip), %ymm3
	vblendvps	%ymm6, LC24(%rip), %ymm3, %ymm3
	vblendvps	%ymm4, %ymm5, %ymm3, %ymm2
	vfmadd132ps	%ymm1, %ymm2, %ymm0
	ret
LFE7:
	.section __TEXT,__text_cold,regular,pure_instructions
LCOLDE26:
	.text
LHOTE26:
	.globl __ZZ5poly3fE1c
	.weak_definition __ZZ5poly3fE1c
	.section __TEXT,__const_coal,coalesced
	.align 4
__ZZ5poly3fE1c:
	.long	1065353216
	.long	3221225472
	.long	1057803469
	.long	1045220557
	.globl __ZZ5poly3fE1b
	.weak_definition __ZZ5poly3fE1b
	.align 4
__ZZ5poly3fE1b:
	.long	1069547520
	.long	1078774989
	.long	3202770534
	.long	1067869798
	.globl __ZZ5poly3fE1a
	.weak_definition __ZZ5poly3fE1a
	.align 4
__ZZ5poly3fE1a:
	.long	3212836864
	.long	1073741824
	.long	1055286886
	.long	3215353446
	.globl _y
	.zerofill __DATA,__pu_bss5,_y,4096,5
	.globl _x
	.zerofill __DATA,__pu_bss5,_x,4096,5
	.literal4
	.align 2
LC0:
	.long	3212836864
	.align 2
LC1:
	.long	1073741824
	.align 2
LC2:
	.long	1055286886
	.align 2
LC3:
	.long	3217031168
	.align 2
LC4:
	.long	1078774989
	.align 2
LC5:
	.long	3205287117
	.align 2
LC6:
	.long	1065353216
	.align 2
LC7:
	.long	3221225472
	.align 2
LC8:
	.long	1057803469
	.align 2
LC9:
	.long	1069547520
	.align 2
LC10:
	.long	3202770534
	.const
	.align 5
LC13:
	.long	1055286886
	.long	1055286886
	.long	1055286886
	.long	1055286886
	.long	1055286886
	.long	1055286886
	.long	1055286886
	.long	1055286886
	.align 5
LC14:
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.align 5
LC15:
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.long	3212836864
	.align 5
LC16:
	.long	3202770534
	.long	3202770534
	.long	3202770534
	.long	3202770534
	.long	3202770534
	.long	3202770534
	.long	3202770534
	.long	3202770534
	.align 5
LC17:
	.long	1078774989
	.long	1078774989
	.long	1078774989
	.long	1078774989
	.long	1078774989
	.long	1078774989
	.long	1078774989
	.long	1078774989
	.align 5
LC18:
	.long	1069547520
	.long	1069547520
	.long	1069547520
	.long	1069547520
	.long	1069547520
	.long	1069547520
	.long	1069547520
	.long	1069547520
	.align 5
LC21:
	.long	3221225472
	.long	3221225472
	.long	3221225472
	.long	3221225472
	.long	3221225472
	.long	3221225472
	.long	3221225472
	.long	3221225472
	.align 5
LC22:
	.long	1057803469
	.long	1057803469
	.long	1057803469
	.long	1057803469
	.long	1057803469
	.long	1057803469
	.long	1057803469
	.long	1057803469
	.align 5
LC23:
	.long	3205287117
	.long	3205287117
	.long	3205287117
	.long	3205287117
	.long	3205287117
	.long	3205287117
	.long	3205287117
	.long	3205287117
	.align 5
LC24:
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.long	1065353216
	.align 5
LC25:
	.long	3217031168
	.long	3217031168
	.long	3217031168
	.long	3217031168
	.long	3217031168
	.long	3217031168
	.long	3217031168
	.long	3217031168
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
	.quad	LFB4-.
	.set L$set$2,LFE4-LFB4
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB5-.
	.set L$set$4,LFE5-LFB5
	.quad L$set$4
	.byte	0
	.byte	0x4
	.set L$set$5,LCFI0-LFB5
	.long L$set$5
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$6,LCFI1-LCFI0
	.long L$set$6
	.byte	0x10
	.byte	0x6
	.byte	0x2
	.byte	0x76
	.byte	0
	.byte	0x4
	.set L$set$7,LCFI2-LCFI1
	.long L$set$7
	.byte	0xf
	.byte	0x3
	.byte	0x76
	.byte	0x78
	.byte	0x6
	.byte	0x4
	.set L$set$8,LCFI3-LCFI2
	.long L$set$8
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$9,LCFI4-LCFI3
	.long L$set$9
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$10,LEFDE5-LASFDE5
	.long L$set$10
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB6-.
	.set L$set$11,LFE6-LFB6
	.quad L$set$11
	.byte	0
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$12,LEFDE7-LASFDE7
	.long L$set$12
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB7-.
	.set L$set$13,LFE7-LFB7
	.quad L$set$13
	.byte	0
	.byte	0x4
	.set L$set$14,LCFI5-LFB7
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
