	.text
	.align 4,0x90
	.globl __Z10contains_vPKfPii
__Z10contains_vPKfPii:
LFB222:
	testl	%edx, %edx
	jle	L15
	pushq	%rbp
LCFI0:
	movq	%rsp, %rbp
LCFI1:
	pushq	%r14
	pushq	%r12
	pushq	%rbx
LCFI2:
	movl	%edx, %ebx
	andq	$-32, %rsp
	shrl	$3, %ebx
	leal	0(,%rbx,8), %r8d
	addq	$24, %rsp
	testl	%r8d, %r8d
	vmovss	_origin(%rip), %xmm5
	vmovss	4+_origin(%rip), %xmm4
	vmovss	8+_origin(%rip), %xmm3
	vmovss	%xmm5, -28(%rsp)
	vmovss	12+_origin(%rip), %xmm7
	vmovss	%xmm4, -32(%rsp)
	vmovss	12+_boxsize(%rip), %xmm0
	vmovss	%xmm3, -36(%rsp)
	movl	_boxsize(%rip), %r9d
	vmovss	%xmm7, -40(%rsp)
	movl	4+_boxsize(%rip), %r10d
	vmovd	%xmm0, %r14d
	movl	8+_boxsize(%rip), %r11d
	je	L9
	cmpl	$7, %edx
	jbe	L9
	vbroadcastss	%xmm5, %ymm13
	vmovd	%r9d, %xmm5
	vbroadcastss	%xmm3, %ymm9
	vbroadcastss	%xmm5, %ymm12
	vmovd	%r10d, %xmm5
	vbroadcastss	%xmm0, %ymm15
	vbroadcastss	%xmm5, %ymm10
	vmovd	%r11d, %xmm5
	vbroadcastss	%xmm4, %ymm11
	vbroadcastss	%xmm5, %ymm8
	vmovaps	LC0(%rip), %ymm1
	movq	%rsi, %rcx
	vbroadcastss	%xmm7, %ymm5
	vmovdqa	LC1(%rip), %ymm0
	movq	%rdi, %rax
	vmovaps	%ymm5, -88(%rsp)
	xorl	%r12d, %r12d
	vmovaps	%ymm9, %ymm14
L8:
	vmovaps	(%rax), %ymm6
	addl	$1, %r12d
	addq	$32, %rcx
	vmovaps	64(%rax), %ymm9
	subq	$-128, %rax
	vmovaps	-96(%rax), %ymm4
	vmovaps	-32(%rax), %ymm3
	vshufps	$136, %ymm4, %ymm6, %ymm5
	vshufps	$221, %ymm4, %ymm6, %ymm4
	vperm2f128	$3, %ymm5, %ymm5, %ymm2
	vshufps	$68, %ymm2, %ymm5, %ymm7
	vshufps	$238, %ymm2, %ymm5, %ymm2
	vinsertf128	$1, %xmm2, %ymm7, %ymm7
	vperm2f128	$3, %ymm4, %ymm4, %ymm2
	vshufps	$68, %ymm2, %ymm4, %ymm5
	vshufps	$238, %ymm2, %ymm4, %ymm2
	vshufps	$136, %ymm3, %ymm9, %ymm4
	vinsertf128	$1, %xmm2, %ymm5, %ymm5
	vshufps	$221, %ymm3, %ymm9, %ymm3
	vperm2f128	$3, %ymm4, %ymm4, %ymm2
	vshufps	$68, %ymm2, %ymm4, %ymm6
	vshufps	$238, %ymm2, %ymm4, %ymm2
	vinsertf128	$1, %xmm2, %ymm6, %ymm6
	vperm2f128	$3, %ymm3, %ymm3, %ymm2
	vshufps	$68, %ymm2, %ymm3, %ymm4
	vshufps	$238, %ymm2, %ymm3, %ymm2
	vshufps	$136, %ymm6, %ymm7, %ymm3
	vinsertf128	$1, %xmm2, %ymm4, %ymm4
	vperm2f128	$3, %ymm3, %ymm3, %ymm2
	vshufps	$68, %ymm2, %ymm3, %ymm9
	vshufps	$238, %ymm2, %ymm3, %ymm2
	vinsertf128	$1, %xmm2, %ymm9, %ymm2
	vsubps	%ymm13, %ymm2, %ymm9
	vshufps	$221, %ymm6, %ymm7, %ymm6
	vandps	%ymm1, %ymm9, %ymm9
	vcmpltps	%ymm12, %ymm9, %ymm9
	vpand	%ymm9, %ymm0, %ymm9
	vpaddd	-32(%rcx), %ymm9, %ymm3
	vmovdqa	%ymm3, -120(%rsp)
	vshufps	$136, %ymm4, %ymm5, %ymm3
	vshufps	$221, %ymm4, %ymm5, %ymm4
	vperm2f128	$3, %ymm3, %ymm3, %ymm2
	vshufps	$68, %ymm2, %ymm3, %ymm9
	vshufps	$238, %ymm2, %ymm3, %ymm2
	vinsertf128	$1, %xmm2, %ymm9, %ymm2
	vsubps	%ymm11, %ymm2, %ymm2
	vandps	%ymm1, %ymm2, %ymm2
	vcmpltps	%ymm10, %ymm2, %ymm2
	vpand	%ymm2, %ymm0, %ymm2
	vpaddd	-120(%rsp), %ymm2, %ymm9
	vperm2f128	$3, %ymm6, %ymm6, %ymm2
	vshufps	$68, %ymm2, %ymm6, %ymm3
	vshufps	$238, %ymm2, %ymm6, %ymm2
	vinsertf128	$1, %xmm2, %ymm3, %ymm2
	vsubps	%ymm14, %ymm2, %ymm3
	vperm2f128	$3, %ymm4, %ymm4, %ymm2
	vshufps	$68, %ymm2, %ymm4, %ymm5
	vshufps	$238, %ymm2, %ymm4, %ymm2
	vinsertf128	$1, %xmm2, %ymm5, %ymm2
	vsubps	-88(%rsp), %ymm2, %ymm2
	vandps	%ymm1, %ymm3, %ymm3
	vcmpltps	%ymm8, %ymm3, %ymm3
	vandps	%ymm1, %ymm2, %ymm2
	vcmpltps	%ymm15, %ymm2, %ymm2
	vpand	%ymm3, %ymm0, %ymm3
	vpaddd	%ymm3, %ymm9, %ymm3
	vpand	%ymm2, %ymm0, %ymm2
	vpaddd	%ymm2, %ymm3, %ymm2
	vmovdqa	%ymm2, -32(%rcx)
	cmpl	%ebx, %r12d
	jb	L8
	cmpl	%edx, %r8d
	je	L17
	vzeroupper
L3:
	leal	0(,%r8,4), %ecx
	vmovss	-28(%rsp), %xmm2
	vmovss	LC2(%rip), %xmm0
	movslq	%ecx, %rcx
	vmovss	-32(%rsp), %xmm3
	vmovss	-36(%rsp), %xmm4
	leaq	(%rdi,%rcx,4), %rax
	addq	%rsi, %rcx
	vmovss	-40(%rsp), %xmm5
	.align 4,0x90
L7:
	vmovss	(%rax), %xmm1
	vmovd	%r9d, %xmm7
	xorl	%esi, %esi
	vmovd	%r10d, %xmm6
	vsubss	%xmm2, %xmm1, %xmm1
	vandps	%xmm0, %xmm1, %xmm1
	vcomiss	%xmm1, %xmm7
	vmovss	4(%rax), %xmm1
	vmovd	%r11d, %xmm7
	vsubss	%xmm3, %xmm1, %xmm1
	seta	%sil
	xorl	%edi, %edi
	addl	(%rcx), %esi
	vandps	%xmm0, %xmm1, %xmm1
	vcomiss	%xmm1, %xmm6
	vmovss	8(%rax), %xmm1
	vmovd	%r14d, %xmm6
	vsubss	%xmm4, %xmm1, %xmm1
	seta	%dil
	addl	%edi, %esi
	xorl	%edi, %edi
	vandps	%xmm0, %xmm1, %xmm1
	vcomiss	%xmm1, %xmm7
	vmovss	12(%rax), %xmm1
	vsubss	%xmm5, %xmm1, %xmm1
	seta	%dil
	addl	%edi, %esi
	xorl	%edi, %edi
	vandps	%xmm0, %xmm1, %xmm1
	vcomiss	%xmm1, %xmm6
	seta	%dil
	addl	$1, %r8d
	addq	$16, %rax
	addl	%edi, %esi
	addq	$4, %rcx
	movl	%esi, -4(%rcx)
	cmpl	%r8d, %edx
	jg	L7
	leaq	-24(%rbp), %rsp
	popq	%rbx
LCFI3:
	popq	%r12
LCFI4:
	popq	%r14
LCFI5:
	popq	%rbp
LCFI6:
L15:
	rep; ret
	.align 4,0x90
L9:
LCFI7:
	xorl	%r8d, %r8d
	jmp	L3
	.align 4,0x90
L17:
	vzeroupper
	leaq	-24(%rbp), %rsp
	popq	%rbx
LCFI8:
	popq	%r12
LCFI9:
	popq	%r14
LCFI10:
	popq	%rbp
LCFI11:
	jmp	L15
LFE222:
	.align 4,0x90
	.globl __Z10contains_bPKfPbi
__Z10contains_bPKfPbi:
LFB223:
	testl	%edx, %edx
	jle	L21
	subl	$1, %edx
	vmovss	_origin(%rip), %xmm5
	vmovss	LC2(%rip), %xmm0
	vmovss	_boxsize(%rip), %xmm9
	leaq	1(%rsi,%rdx), %rcx
	vmovss	4+_origin(%rip), %xmm4
	vmovss	4+_boxsize(%rip), %xmm8
	vmovss	8+_origin(%rip), %xmm3
	vmovss	8+_boxsize(%rip), %xmm7
	vmovss	12+_origin(%rip), %xmm2
	vmovss	12+_boxsize(%rip), %xmm6
	.align 4,0x90
L20:
	vmovss	(%rdi), %xmm1
	vsubss	%xmm5, %xmm1, %xmm1
	vandps	%xmm0, %xmm1, %xmm1
	vcomiss	%xmm1, %xmm9
	vmovss	4(%rdi), %xmm1
	vsubss	%xmm4, %xmm1, %xmm1
	seta	%al
	vandps	%xmm0, %xmm1, %xmm1
	vcomiss	%xmm1, %xmm8
	vmovss	8(%rdi), %xmm1
	vsubss	%xmm3, %xmm1, %xmm1
	seta	%dl
	andl	%edx, %eax
	andb	(%rsi), %al
	vandps	%xmm0, %xmm1, %xmm1
	vcomiss	%xmm1, %xmm7
	vmovss	12(%rdi), %xmm1
	vsubss	%xmm2, %xmm1, %xmm1
	seta	%dl
	andl	%edx, %eax
	vandps	%xmm0, %xmm1, %xmm1
	vcomiss	%xmm1, %xmm6
	seta	%dl
	addq	$1, %rsi
	addq	$16, %rdi
	andl	%edx, %eax
	movb	%al, -1(%rsi)
	cmpq	%rcx, %rsi
	jne	L20
L21:
	rep; ret
LFE223:
	.align 4,0x90
	.globl __Z9containsDv
__Z9containsDv:
LFB225:
	vmovapd	_ori(%rip), %ymm0
	leaq	_pointsD(%rip), %rax
	vmovapd	_s(%rip), %ymm3
	leaq	_ins(%rip), %rdx
	leaq	32768+_pointsD(%rip), %rdi
	vsubpd	%ymm3, %ymm0, %ymm4
	vaddpd	%ymm0, %ymm3, %ymm3
	.align 4,0x90
L23:
	vmovapd	(%rax), %ymm1
	vcmpltpd	%ymm3, %ymm1, %ymm2
	vcmpltpd	%ymm1, %ymm4, %ymm0
	vpand	%ymm2, %ymm0, %ymm0
	vmovdqa	%xmm0, %xmm1
	vextracti128	$0x1, %ymm0, %xmm0
	vmovd	%xmm1, %rsi
	vpextrq	$1, %xmm1, %rcx
	addq	%rsi, %rcx
	vmovd	%xmm0, %rsi
	addq	%rsi, %rcx
	cmpq	$3, %rcx
	sete	(%rdx)
	addq	$32, %rax
	addq	$1, %rdx
	cmpq	%rdi, %rax
	jne	L23
	vzeroupper
	ret
LFE225:
	.align 4,0x90
	.globl __Z8containsU8__vectorf
__Z8containsU8__vectorf:
LFB226:
	vmovaps	_orif(%rip), %xmm2
	vmovaps	_sf(%rip), %xmm1
	vaddps	%xmm2, %xmm1, %xmm3
	vsubps	%xmm1, %xmm2, %xmm1
	vcmpltps	%xmm3, %xmm0, %xmm3
	vcmpltps	%xmm0, %xmm1, %xmm0
	vpand	%xmm3, %xmm0, %xmm0
	vmovd	%xmm0, %edx
	vpextrd	$1, %xmm0, %eax
	addl	%edx, %eax
	vpextrd	$2, %xmm0, %edx
	addl	%edx, %eax
	cmpl	$3, %eax
	sete	%al
	ret
LFE226:
	.align 4,0x90
	.globl __Z9containsFv
__Z9containsFv:
LFB227:
	vmovaps	_orif(%rip), %xmm0
	leaq	_pointsF(%rip), %rax
	vmovaps	_sf(%rip), %xmm3
	leaq	_ins(%rip), %rdx
	leaq	16384+_pointsF(%rip), %rdi
	vsubps	%xmm3, %xmm0, %xmm4
	vaddps	%xmm0, %xmm3, %xmm3
	.align 4,0x90
L27:
	vmovaps	(%rax), %xmm1
	vcmpltps	%xmm3, %xmm1, %xmm2
	vcmpltps	%xmm1, %xmm4, %xmm0
	vpand	%xmm2, %xmm0, %xmm0
	vmovd	%xmm0, %esi
	vpextrd	$1, %xmm0, %ecx
	addl	%esi, %ecx
	vpextrd	$2, %xmm0, %esi
	addl	%esi, %ecx
	cmpl	$3, %ecx
	sete	(%rdx)
	addq	$16, %rax
	addq	$1, %rdx
	cmpq	%rdi, %rax
	jne	L27
	rep; ret
LFE227:
	.globl _pointsF
	.zerofill __DATA,__pu_bss5,_pointsF,16384,5
	.globl _sf
	.zerofill __DATA,__pu_bss4,_sf,16,4
	.globl _orif
	.zerofill __DATA,__pu_bss4,_orif,16,4
	.globl _ins
	.zerofill __DATA,__pu_bss5,_ins,1024,5
	.globl _pointsD
	.zerofill __DATA,__pu_bss5,_pointsD,32768,5
	.globl _s
	.zerofill __DATA,__pu_bss5,_s,32,5
	.globl _ori
	.zerofill __DATA,__pu_bss5,_ori,32,5
	.globl _boxsize
	.zerofill __DATA,__pu_bss4,_boxsize,16,4
	.globl _origin
	.zerofill __DATA,__pu_bss4,_origin,16,4
	.const
	.align 5
LC0:
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.long	2147483647
	.align 5
LC1:
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.long	1
	.literal16
	.align 4
LC2:
	.long	2147483647
	.long	0
	.long	0
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
	.quad	LFB222-.
	.set L$set$2,LFE222-LFB222
	.quad L$set$2
	.byte	0
	.byte	0x4
	.set L$set$3,LCFI0-LFB222
	.long L$set$3
	.byte	0xe
	.byte	0x10
	.byte	0x86
	.byte	0x2
	.byte	0x4
	.set L$set$4,LCFI1-LCFI0
	.long L$set$4
	.byte	0xd
	.byte	0x6
	.byte	0x4
	.set L$set$5,LCFI2-LCFI1
	.long L$set$5
	.byte	0x8e
	.byte	0x3
	.byte	0x8c
	.byte	0x4
	.byte	0x83
	.byte	0x5
	.byte	0x4
	.set L$set$6,LCFI3-LCFI2
	.long L$set$6
	.byte	0xc3
	.byte	0x4
	.set L$set$7,LCFI4-LCFI3
	.long L$set$7
	.byte	0xcc
	.byte	0x4
	.set L$set$8,LCFI5-LCFI4
	.long L$set$8
	.byte	0xce
	.byte	0x4
	.set L$set$9,LCFI6-LCFI5
	.long L$set$9
	.byte	0xc6
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.byte	0x4
	.set L$set$10,LCFI7-LCFI6
	.long L$set$10
	.byte	0xc
	.byte	0x6
	.byte	0x10
	.byte	0x83
	.byte	0x5
	.byte	0x86
	.byte	0x2
	.byte	0x8c
	.byte	0x4
	.byte	0x8e
	.byte	0x3
	.byte	0x4
	.set L$set$11,LCFI8-LCFI7
	.long L$set$11
	.byte	0xc3
	.byte	0x4
	.set L$set$12,LCFI9-LCFI8
	.long L$set$12
	.byte	0xcc
	.byte	0x4
	.set L$set$13,LCFI10-LCFI9
	.long L$set$13
	.byte	0xce
	.byte	0x4
	.set L$set$14,LCFI11-LCFI10
	.long L$set$14
	.byte	0xc6
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$15,LEFDE3-LASFDE3
	.long L$set$15
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB223-.
	.set L$set$16,LFE223-LFB223
	.quad L$set$16
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$17,LEFDE5-LASFDE5
	.long L$set$17
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB225-.
	.set L$set$18,LFE225-LFB225
	.quad L$set$18
	.byte	0
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$19,LEFDE7-LASFDE7
	.long L$set$19
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB226-.
	.set L$set$20,LFE226-LFB226
	.quad L$set$20
	.byte	0
	.align 3
LEFDE7:
LSFDE9:
	.set L$set$21,LEFDE9-LASFDE9
	.long L$set$21
LASFDE9:
	.long	LASFDE9-EH_frame1
	.quad	LFB227-.
	.set L$set$22,LFE227-LFB227
	.quad L$set$22
	.byte	0
	.align 3
LEFDE9:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
