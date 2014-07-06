	.text
	.align 4,0x90
	.globl __ZN8justcomp3barEv
__ZN8justcomp3barEv:
LFB2105:
	vmovaps	LC0(%rip), %xmm12
	xorl	%eax, %eax
	vmovaps	LC1(%rip), %xmm11
	leaq	__ZN8justcomp1aE(%rip), %rcx
	vmovaps	LC2(%rip), %xmm10
	leaq	__ZN8justcomp1bE(%rip), %rdx
	vmovaps	LC3(%rip), %xmm9
	vmovaps	LC4(%rip), %xmm3
	vmovaps	LC5(%rip), %xmm8
	vmovaps	LC6(%rip), %xmm7
	vmovaps	LC7(%rip), %xmm6
	vmovaps	LC8(%rip), %xmm5
	vmovdqa	LC9(%rip), %xmm4
	.align 4,0x90
L3:
	vmovaps	(%rcx,%rax), %xmm0
	vmulps	%xmm12, %xmm0, %xmm1
	vaddps	%xmm11, %xmm1, %xmm1
	vroundps	$1, %xmm1, %xmm1
	vmulps	%xmm10, %xmm1, %xmm2
	vsubps	%xmm2, %xmm0, %xmm0
	vmulps	%xmm9, %xmm1, %xmm2
	vcvttps2dq	%xmm1, %xmm1
	vpaddd	%xmm4, %xmm1, %xmm1
	vpslld	$23, %xmm1, %xmm1
	vsubps	%xmm2, %xmm0, %xmm0
	vmulps	%xmm6, %xmm0, %xmm14
	vaddps	%xmm3, %xmm0, %xmm13
	vmulps	%xmm0, %xmm0, %xmm2
	vmulps	%xmm8, %xmm0, %xmm15
	vaddps	%xmm5, %xmm14, %xmm14
	vaddps	%xmm7, %xmm15, %xmm15
	vmulps	%xmm2, %xmm14, %xmm14
	vaddps	%xmm14, %xmm15, %xmm14
	vmulps	%xmm2, %xmm14, %xmm2
	vaddps	%xmm2, %xmm13, %xmm2
	vmulps	%xmm0, %xmm2, %xmm0
	vaddps	%xmm3, %xmm0, %xmm0
	vmulps	%xmm1, %xmm0, %xmm0
	vmovaps	%xmm0, (%rdx,%rax)
	addq	$16, %rax
	cmpq	$4194304, %rax
	jne	L3
	rep; ret
LFE2105:
	.cstring
LC14:
	.ascii "time \0"
LC15:
	.ascii "sum=\0"
LC16:
	.ascii "to prevent compiler optim.\0"
	.section __TEXT,__text_startup,regular,pure_instructions
	.align 4
	.globl _main
_main:
LFB2943:
	pushq	%rbp
LCFI0:
	vxorps	%xmm4, %xmm4, %xmm4
	xorl	%r11d, %r11d
	leaq	__ZN8justcomp1aE(%rip), %r8
	movq	%rsp, %rbp
LCFI1:
	pushq	%rbx
	leaq	__ZN8justcomp1bE(%rip), %rdi
	andq	$-32, %rsp
	xorl	%r10d, %r10d
	leaq	4194304+__ZN8justcomp1aE(%rip), %rsi
	subq	$16, %rsp
LCFI2:
	vmovss	%xmm4, 12(%rsp)
	vmovss	LC10(%rip), %xmm8
	vmovss	LC12(%rip), %xmm1
	vmovaps	LC0(%rip), %xmm15
	leaq	4194304+__ZN8justcomp1bE(%rip), %r9
	vmovaps	LC1(%rip), %xmm14
	vmovaps	LC2(%rip), %xmm13
	vmovaps	LC3(%rip), %xmm12
	vmovaps	LC4(%rip), %xmm4
	vmovaps	LC5(%rip), %xmm11
	vmovaps	LC6(%rip), %xmm10
	vmovaps	LC7(%rip), %xmm9
	.align 4
L14:
	leaq	__ZN8justcomp1aE(%rip), %rax
	addq	$1, %r11
	.align 4
L7:
	vmovd	%xmm8, %ecx
	addq	$8, %rax
	addl	$1, %ecx
	movl	%ecx, -8(%rax)
	vmovd	%ecx, %xmm0
	vmovd	%ecx, %xmm8
	vxorps	%xmm1, %xmm0, %xmm0
	vmovss	%xmm0, -4(%rax)
	cmpq	%rsi, %rax
	jne	L7
	rdtsc
	salq	$32, %rdx
	movq	%rax, %rbx
	orq	%rdx, %rbx
	xorl	%edx, %edx
	.align 4
L9:
	vmovaps	(%r8,%rdx), %xmm0
	vmulps	%xmm15, %xmm0, %xmm2
	vaddps	%xmm14, %xmm2, %xmm2
	vroundps	$1, %xmm2, %xmm2
	vmulps	%xmm13, %xmm2, %xmm3
	vsubps	%xmm3, %xmm0, %xmm0
	vmulps	%xmm12, %xmm2, %xmm3
	vcvttps2dq	%xmm2, %xmm2
	vpaddd	LC9(%rip), %xmm2, %xmm2
	vpslld	$23, %xmm2, %xmm2
	vsubps	%xmm3, %xmm0, %xmm0
	vmulps	%xmm9, %xmm0, %xmm6
	vaddps	%xmm4, %xmm0, %xmm5
	vmulps	%xmm0, %xmm0, %xmm3
	vmulps	%xmm11, %xmm0, %xmm7
	vaddps	LC8(%rip), %xmm6, %xmm6
	vaddps	%xmm10, %xmm7, %xmm7
	vmulps	%xmm3, %xmm6, %xmm6
	vaddps	%xmm6, %xmm7, %xmm6
	vmulps	%xmm3, %xmm6, %xmm3
	vaddps	%xmm3, %xmm5, %xmm3
	vmulps	%xmm0, %xmm3, %xmm0
	vaddps	%xmm4, %xmm0, %xmm0
	vmulps	%xmm2, %xmm0, %xmm0
	vmovaps	%xmm0, (%rdi,%rdx)
	addq	$16, %rdx
	cmpq	$4194304, %rdx
	jne	L9
	rdtsc
	vxorps	%xmm0, %xmm0, %xmm0
	salq	$32, %rdx
	orq	%rax, %rdx
	leaq	__ZN8justcomp1bE(%rip), %rax
	subq	%rbx, %rdx
	addq	%rdx, %r10
	.align 4
L11:
	vaddps	(%rax), %ymm0, %ymm0
	addq	$32, %rax
	cmpq	%r9, %rax
	jne	L11
	vhaddps	%ymm0, %ymm0, %ymm0
	vhaddps	%ymm0, %ymm0, %ymm2
	vperm2f128	$1, %ymm2, %ymm2, %ymm0
	vaddps	%ymm2, %ymm0, %ymm0
	vaddss	12(%rsp), %xmm0, %xmm5
	vmovss	%xmm5, 12(%rsp)
	vmovss	LC13(%rip), %xmm5
	vcomiss	%xmm8, %xmm5
	ja	L14
	vcvtsi2sdq	%r10, %xmm0, %xmm0
	testq	%r10, %r10
	js	L19
L16:
	salq	$20, %r11
	movq	__ZSt4cout@GOTPCREL(%rip), %rbx
	vcvtsi2sdq	%r11, %xmm1, %xmm1
	leaq	LC14(%rip), %rsi
	movq	%rbx, %rdi
	vdivsd	%xmm1, %xmm0, %xmm0
	vmovsd	%xmm0, (%rsp)
	vzeroupper
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	vmovsd	(%rsp), %xmm0
	movq	%rax, %rdi
	call	__ZNSo9_M_insertIdEERSoT_
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	leaq	LC15(%rip), %rsi
	movq	%rbx, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	vmovss	12(%rsp), %xmm0
	movq	%rax, %rdi
	vcvtps2pd	%xmm0, %xmm0
	call	__ZNSo9_M_insertIdEERSoT_
	leaq	LC16(%rip), %rsi
	movq	%rax, %rdi
	call	__ZStlsISt11char_traitsIcEERSt13basic_ostreamIcT_ES5_PKc
	movq	%rax, %rdi
	call	__ZSt4endlIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_
	xorl	%eax, %eax
	movq	-8(%rbp), %rbx
	leave
LCFI3:
	ret
L19:
LCFI4:
	movq	%r10, %rax
	andl	$1, %r10d
	shrq	%rax
	orq	%r10, %rax
	vcvtsi2sdq	%rax, %xmm0, %xmm0
	vaddsd	%xmm0, %xmm0, %xmm0
	jmp	L16
LFE2943:
	.align 4
__GLOBAL__sub_I_dirtyexp.cc:
LFB3108:
	leaq	__ZStL8__ioinit(%rip), %rdi
	subq	$8, %rsp
LCFI5:
	call	__ZNSt8ios_base4InitC1Ev
	movq	__ZNSt8ios_base4InitD1Ev@GOTPCREL(%rip), %rdi
	addq	$8, %rsp
LCFI6:
	leaq	___dso_handle(%rip), %rdx
	leaq	__ZStL8__ioinit(%rip), %rsi
	jmp	___cxa_atexit
LFE3108:
	.mod_init_func
	.align 3
	.quad	__GLOBAL__sub_I_dirtyexp.cc
	.globl __ZN8justcomp1bE
	.zerofill __DATA,__pu_bss5,__ZN8justcomp1bE,4194304,5
	.globl __ZN8justcomp1aE
	.zerofill __DATA,__pu_bss5,__ZN8justcomp1aE,4194304,5
	.static_data
__ZStL8__ioinit:
	.space	1
	.literal16
	.align 4
LC0:
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.align 4
LC1:
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.align 4
LC2:
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.align 4
LC3:
	.long	901758606
	.long	901758606
	.long	901758606
	.long	901758606
	.align 4
LC4:
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.align 4
LC5:
	.long	1034594926
	.long	1034594926
	.long	1034594926
	.long	1034594926
	.align 4
LC6:
	.long	1051372102
	.long	1051372102
	.long	1051372102
	.long	1051372102
	.align 4
LC7:
	.long	993439040
	.long	993439040
	.long	993439040
	.long	993439040
	.align 4
LC8:
	.long	1015619202
	.long	1015619202
	.long	1015619202
	.long	1015619202
	.align 4
LC9:
	.long	126
	.long	126
	.long	126
	.long	126
	.literal4
	.align 2
LC10:
	.long	1065353216
	.literal16
	.align 4
LC12:
	.long	2147483648
	.long	0
	.long	0
	.long	0
	.literal4
	.align 2
LC13:
	.long	1107296256
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
	.quad	LFB2105-.
	.set L$set$2,LFE2105-LFB2105
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB2943-.
	.set L$set$4,LFE2943-LFB2943
	.quad L$set$4
	.byte	0
	.byte	0x4
	.set L$set$5,LCFI0-LFB2943
	.long L$set$5
	.byte	0xe
	.byte	0x10
	.byte	0x86
	.byte	0x2
	.byte	0x4
	.set L$set$6,LCFI1-LCFI0
	.long L$set$6
	.byte	0xd
	.byte	0x6
	.byte	0x4
	.set L$set$7,LCFI2-LCFI1
	.long L$set$7
	.byte	0x83
	.byte	0x3
	.byte	0x4
	.set L$set$8,LCFI3-LCFI2
	.long L$set$8
	.byte	0xa
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.byte	0x4
	.set L$set$9,LCFI4-LCFI3
	.long L$set$9
	.byte	0xb
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$10,LEFDE5-LASFDE5
	.long L$set$10
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB3108-.
	.set L$set$11,LFE3108-LFB3108
	.quad L$set$11
	.byte	0
	.byte	0x4
	.set L$set$12,LCFI5-LFB3108
	.long L$set$12
	.byte	0xe
	.byte	0x10
	.byte	0x4
	.set L$set$13,LCFI6-LCFI5
	.long L$set$13
	.byte	0xe
	.byte	0x8
	.align 3
LEFDE5:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
