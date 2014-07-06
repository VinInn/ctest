	.text
	.align 4,0x90
	.globl __Z3barPfPKfi
__Z3barPfPKfi:
LFB3792:
	leaq	8(%rsp), %r10
LCFI0:
	andq	$-32, %rsp
	testl	%edx, %edx
	pushq	-8(%r10)
	pushq	%rbp
LCFI1:
	movq	%rsp, %rbp
	pushq	%r10
LCFI2:
	jle	L12
	movl	%edx, %eax
	shrl	$3, %eax
	movl	%eax, %r8d
	sall	$3, %r8d
	je	L9
	cmpl	$7, %edx
	jbe	L9
	vmovaps	LC0(%rip), %ymm14
	xorl	%ecx, %ecx
	xorl	%r9d, %r9d
	vmovaps	LC1(%rip), %ymm13
	vmovaps	LC2(%rip), %ymm12
	vmovaps	LC3(%rip), %ymm11
	vmovaps	LC4(%rip), %ymm10
	vmovaps	LC5(%rip), %ymm9
	vmovaps	LC6(%rip), %ymm8
	vmovaps	LC7(%rip), %ymm7
	vmovaps	LC8(%rip), %ymm6
	vmovaps	LC9(%rip), %ymm5
	vmovaps	LC10(%rip), %ymm3
	vmovdqa	LC11(%rip), %ymm4
L8:
	vmaxps	(%rsi,%rcx), %ymm14, %ymm1
	addl	$1, %r9d
	vminps	%ymm13, %ymm1, %ymm1
	vmovaps	%ymm1, %ymm2
	vfmadd132ps	%ymm12, %ymm11, %ymm2
	vroundps	$1, %ymm2, %ymm2
	vfnmadd231ps	%ymm10, %ymm2, %ymm1
	vfnmadd231ps	%ymm9, %ymm2, %ymm1
	vmovaps	%ymm1, %ymm0
	vfmadd132ps	%ymm8, %ymm7, %ymm0
	vfmadd132ps	%ymm1, %ymm6, %ymm0
	vfmadd132ps	%ymm1, %ymm5, %ymm0
	vfmadd132ps	%ymm1, %ymm3, %ymm0
	vfmadd132ps	%ymm1, %ymm3, %ymm0
	vcvttps2dq	%ymm2, %ymm1
	vpaddd	%ymm4, %ymm1, %ymm1
	vpslld	$23, %ymm1, %ymm1
	vmulps	%ymm1, %ymm0, %ymm0
	vmovaps	%ymm0, (%rdi,%rcx)
	addq	$32, %rcx
	cmpl	%eax, %r9d
	jb	L8
	cmpl	%edx, %r8d
	je	L15
	vzeroupper
L3:
	movslq	%r8d, %rax
	vmovss	LC12(%rip), %xmm13
	salq	$2, %rax
	vmovss	LC13(%rip), %xmm12
	vmovss	LC14(%rip), %xmm11
	addq	%rax, %rsi
	addq	%rax, %rdi
	xorl	%eax, %eax
	vmovss	LC15(%rip), %xmm10
	vmovss	LC16(%rip), %xmm9
	vmovss	LC17(%rip), %xmm8
	vmovss	LC18(%rip), %xmm7
	vmovss	LC19(%rip), %xmm6
	vmovss	LC20(%rip), %xmm5
	vmovss	LC21(%rip), %xmm4
	vmovss	LC22(%rip), %xmm3
	.align 4,0x90
L7:
	vmaxss	(%rsi,%rax,4), %xmm13, %xmm1
	vminss	%xmm12, %xmm1, %xmm1
	vmovaps	%xmm1, %xmm2
	vfmadd132ss	%xmm11, %xmm10, %xmm2
	vroundss	$1, %xmm2, %xmm2, %xmm2
	vcvttss2si	%xmm2, %ecx
	vfnmadd231ss	%xmm9, %xmm2, %xmm1
	vfnmadd231ss	%xmm8, %xmm2, %xmm1
	vmovaps	%xmm1, %xmm0
	vfmadd132ss	%xmm7, %xmm6, %xmm0
	vfmadd132ss	%xmm1, %xmm5, %xmm0
	vfmadd132ss	%xmm1, %xmm4, %xmm0
	vfmadd132ss	%xmm1, %xmm3, %xmm0
	vfmadd132ss	%xmm1, %xmm3, %xmm0
	addl	$126, %ecx
	sall	$23, %ecx
	vmovd	%ecx, %xmm2
	vmulss	%xmm2, %xmm0, %xmm0
	vmovss	%xmm0, (%rdi,%rax,4)
	addq	$1, %rax
	leal	(%r8,%rax), %ecx
	cmpl	%ecx, %edx
	jg	L7
L12:
	popq	%r10
LCFI3:
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI4:
	ret
	.align 4,0x90
L15:
LCFI5:
	vzeroupper
	popq	%r10
LCFI6:
	popq	%rbp
	leaq	-8(%r10), %rsp
LCFI7:
	ret
	.align 4,0x90
L9:
LCFI8:
	xorl	%r8d, %r8d
	jmp	L3
LFE3792:
	.const
	.align 5
LC0:
	.long	3266227280
	.long	3266227280
	.long	3266227280
	.long	3266227280
	.long	3266227280
	.long	3266227280
	.long	3266227280
	.long	3266227280
	.align 5
LC1:
	.long	1118925336
	.long	1118925336
	.long	1118925336
	.long	1118925336
	.long	1118925336
	.long	1118925336
	.long	1118925336
	.long	1118925336
	.align 5
LC2:
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.long	1069066811
	.align 5
LC3:
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.long	1056964608
	.align 5
LC4:
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.long	1060205056
	.align 5
LC5:
	.long	901758606
	.long	901758606
	.long	901758606
	.long	901758606
	.long	901758606
	.long	901758606
	.long	901758606
	.long	901758606
	.align 5
LC6:
	.long	1015621328
	.long	1015621328
	.long	1015621328
	.long	1015621328
	.long	1015621328
	.long	1015621328
	.long	1015621328
	.long	1015621328
	.align 5
LC7:
	.long	1034657900
	.long	1034657900
	.long	1034657900
	.long	1034657900
	.long	1034657900
	.long	1034657900
	.long	1034657900
	.long	1034657900
	.align 5
LC8:
	.long	1051372088
	.long	1051372088
	.long	1051372088
	.long	1051372088
	.long	1051372088
	.long	1051372088
	.long	1051372088
	.long	1051372088
	.align 5
LC9:
	.long	1065352920
	.long	1065352920
	.long	1065352920
	.long	1065352920
	.long	1065352920
	.long	1065352920
	.long	1065352920
	.long	1065352920
	.align 5
LC10:
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.long	1073741824
	.align 5
LC11:
	.long	126
	.long	126
	.long	126
	.long	126
	.long	126
	.long	126
	.long	126
	.long	126
	.literal4
	.align 2
LC12:
	.long	3266227280
	.align 2
LC13:
	.long	1118925336
	.align 2
LC14:
	.long	1069066811
	.align 2
LC15:
	.long	1056964608
	.align 2
LC16:
	.long	1060205056
	.align 2
LC17:
	.long	901758606
	.align 2
LC18:
	.long	1015621328
	.align 2
LC19:
	.long	1034657900
	.align 2
LC20:
	.long	1051372088
	.align 2
LC21:
	.long	1065352920
	.align 2
LC22:
	.long	1073741824
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
	.quad	LFB3792-.
	.set L$set$2,LFE3792-LFB3792
	.quad L$set$2
	.byte	0
	.byte	0x4
	.set L$set$3,LCFI0-LFB3792
	.long L$set$3
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$4,LCFI1-LCFI0
	.long L$set$4
	.byte	0x10
	.byte	0x6
	.byte	0x2
	.byte	0x76
	.byte	0
	.byte	0x4
	.set L$set$5,LCFI2-LCFI1
	.long L$set$5
	.byte	0xf
	.byte	0x3
	.byte	0x76
	.byte	0x78
	.byte	0x6
	.byte	0x4
	.set L$set$6,LCFI3-LCFI2
	.long L$set$6
	.byte	0xa
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$7,LCFI4-LCFI3
	.long L$set$7
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.byte	0x4
	.set L$set$8,LCFI5-LCFI4
	.long L$set$8
	.byte	0xb
	.byte	0x4
	.set L$set$9,LCFI6-LCFI5
	.long L$set$9
	.byte	0xa
	.byte	0xc
	.byte	0xa
	.byte	0
	.byte	0x4
	.set L$set$10,LCFI7-LCFI6
	.long L$set$10
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.byte	0x4
	.set L$set$11,LCFI8-LCFI7
	.long L$set$11
	.byte	0xb
	.align 3
LEFDE1:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
