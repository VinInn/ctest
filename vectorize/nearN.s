	.text
	.align 4,0x90
	.globl __Z6nearNSi
__Z6nearNSi:
LFB3773:
	leaq	_nn(%rip), %r8
	movslq	%edi, %rdi
	xorl	%eax, %eax
	movl	(%r8,%rdi,4), %esi
	leaq	_distNN(%rip), %r9
	leaq	_phi(%rip), %rcx
	movss	(%r9,%rdi,4), %xmm2
	leaq	_eta(%rip), %rdx
	movss	(%rcx,%rdi,4), %xmm4
	movss	(%rdx,%rdi,4), %xmm3
	.align 4,0x90
L2:
	movss	(%rcx,%rax), %xmm0
	movss	(%rdx,%rax), %xmm1
	subss	%xmm4, %xmm0
	subss	%xmm3, %xmm1
	mulss	%xmm0, %xmm0
	mulss	%xmm1, %xmm1
	addss	%xmm1, %xmm0
	comiss	%xmm0, %xmm2
	jbe	L3
	leaq	_index(%rip), %rsi
	movl	(%rsi,%rax), %esi
L3:
	addq	$4, %rax
	minss	%xmm0, %xmm2
	cmpq	$4096, %rax
	jne	L2
	movss	%xmm2, (%r9,%rdi,4)
	movl	%esi, (%r8,%rdi,4)
	ret
LFE3773:
	.align 4,0x90
	.globl __Z5nearNi
__Z5nearNi:
LFB3774:
	movslq	%edi, %rax
	movss	LC1(%rip), %xmm3
	movss	LC3(%rip), %xmm6
	leaq	_phi(%rip), %r10
	movss	LC0(%rip), %xmm7
	leaq	_eta(%rip), %r9
	movss	(%r10,%rax,4), %xmm5
	movss	(%r9,%rax,4), %xmm4
	leaq	_distNN(%rip), %rcx
	xorl	%eax, %eax
	leaq	_nn(%rip), %r8
	jmp	L9
	.align 4,0x90
L22:
	mulss	%xmm0, %xmm0
	mulss	%xmm1, %xmm1
	addss	%xmm1, %xmm0
L12:
	comiss	%xmm2, %xmm0
	movl	%edi, %esi
	jb	L13
	movl	(%r8,%rdx), %esi
L13:
	movaps	%xmm2, %xmm1
	addq	$1, %rax
	movl	%esi, (%r8,%rdx)
	minss	%xmm0, %xmm1
	cmpq	$1024, %rax
	movss	%xmm1, (%rcx,%rdx)
	je	L21
L9:
	movss	(%r10,%rax,4), %xmm0
	leaq	0(,%rax,4), %rdx
	movss	(%r9,%rax,4), %xmm1
	subss	%xmm5, %xmm0
	movss	(%rcx,%rax,4), %xmm2
	subss	%xmm4, %xmm1
	andps	%xmm3, %xmm0
	comiss	LC2(%rip), %xmm0
	jbe	L10
	movaps	%xmm6, %xmm8
	subss	%xmm0, %xmm8
	movaps	%xmm8, %xmm0
L10:
	cmpl	%eax, %edi
	jne	L22
	movaps	%xmm7, %xmm0
	jmp	L12
	.align 4,0x90
L21:
	ret
LFE3774:
	.align 4,0x90
	.globl __Z6nearNIi
__Z6nearNIi:
LFB3775:
	movslq	%edi, %r9
	movss	LC3(%rip), %xmm7
	xorl	%eax, %eax
	movss	LC1(%rip), %xmm8
	leaq	_nn(%rip), %r10
	movss	LC2(%rip), %xmm6
	movl	(%r10,%r9,4), %edx
	leaq	_distNN(%rip), %r11
	leaq	_phi(%rip), %r8
	movss	(%r11,%r9,4), %xmm4
	leaq	_eta(%rip), %rsi
	movss	(%r8,%r9,4), %xmm10
	movss	(%rsi,%r9,4), %xmm9
	movss	LC0(%rip), %xmm5
	.align 4,0x90
L24:
	movss	(%r8,%rax,4), %xmm0
	movaps	%xmm7, %xmm3
	movaps	%xmm6, %xmm2
	cmpl	%eax, %edi
	movss	(%rsi,%rax,4), %xmm1
	movl	%eax, %ecx
	subss	%xmm10, %xmm0
	subss	%xmm9, %xmm1
	andps	%xmm8, %xmm0
	cmpltss	%xmm0, %xmm2
	subss	%xmm0, %xmm3
	andps	%xmm2, %xmm3
	andnps	%xmm0, %xmm2
	movaps	%xmm5, %xmm0
	orps	%xmm3, %xmm2
	je	L26
	mulss	%xmm1, %xmm1
	movaps	%xmm2, %xmm0
	mulss	%xmm2, %xmm0
	addss	%xmm1, %xmm0
L26:
	comiss	%xmm0, %xmm4
	minss	%xmm0, %xmm4
	cmova	%ecx, %edx
	addq	$1, %rax
	cmpq	$1024, %rax
	jne	L24
	movss	%xmm4, (%r11,%r9,4)
	movl	%edx, (%r10,%r9,4)
	ret
LFE3775:
	.align 4,0x90
	.globl __Z6nearNOi
__Z6nearNOi:
LFB3776:
	movslq	%edi, %r9
	xorl	%eax, %eax
	movss	LC1(%rip), %xmm8
	movss	LC3(%rip), %xmm7
	leaq	_distNN(%rip), %r11
	movss	LC2(%rip), %xmm6
	leaq	_nn(%rip), %r10
	movss	(%r11,%r9,4), %xmm4
	leaq	_phi(%rip), %r8
	movl	(%r10,%r9,4), %edx
	leaq	_eta(%rip), %rsi
	movss	(%r8,%r9,4), %xmm10
	movss	(%rsi,%r9,4), %xmm9
	movss	LC0(%rip), %xmm5
	jmp	L29
	.align 4,0x90
L35:
	movaps	%xmm1, %xmm4
L29:
	movss	(%r8,%rax,4), %xmm0
	movaps	%xmm7, %xmm3
	movaps	%xmm6, %xmm2
	cmpl	%eax, %edi
	movss	(%rsi,%rax,4), %xmm1
	movl	%eax, %ecx
	subss	%xmm10, %xmm0
	subss	%xmm9, %xmm1
	andps	%xmm8, %xmm0
	cmpltss	%xmm0, %xmm2
	subss	%xmm0, %xmm3
	andps	%xmm2, %xmm3
	andnps	%xmm0, %xmm2
	movaps	%xmm5, %xmm0
	orps	%xmm3, %xmm2
	je	L31
	mulss	%xmm1, %xmm1
	movaps	%xmm2, %xmm0
	mulss	%xmm2, %xmm0
	addss	%xmm1, %xmm0
L31:
	comiss	%xmm0, %xmm4
	movaps	%xmm0, %xmm1
	minss	%xmm4, %xmm1
	cmova	%ecx, %edx
	addq	$1, %rax
	cmpq	$1024, %rax
	jne	L35
	movss	%xmm1, (%r11,%r9,4)
	movl	%edx, (%r10,%r9,4)
	ret
LFE3776:
	.align 4,0x90
	.globl __Z6nearN2i
__Z6nearN2i:
LFB3778:
	movss	LC3(%rip), %xmm4
	xorl	%eax, %eax
	movslq	%edi, %r8
	movss	LC1(%rip), %xmm3
	leaq	_distNN(%rip), %rsi
	leaq	_nn(%rip), %r9
	.align 4,0x90
L37:
	leaq	0(,%rax,4), %rdx
	cmpl	%eax, %edi
	movss	(%rsi,%rax,4), %xmm2
	je	L44
	leaq	_phi(%rip), %rcx
	movss	(%rcx,%rax,4), %xmm0
	subss	(%rcx,%r8,4), %xmm0
	leaq	_eta(%rip), %rcx
	movss	(%rcx,%rax,4), %xmm1
	subss	(%rcx,%r8,4), %xmm1
	andps	%xmm3, %xmm0
	comiss	LC2(%rip), %xmm0
	jbe	L39
	movaps	%xmm4, %xmm5
	subss	%xmm0, %xmm5
	movaps	%xmm5, %xmm0
L39:
	mulss	%xmm0, %xmm0
	mulss	%xmm1, %xmm1
	addss	%xmm1, %xmm0
L38:
	comiss	%xmm2, %xmm0
	movl	%edi, %ecx
	jb	L41
	movl	(%r9,%rdx), %ecx
L41:
	movaps	%xmm2, %xmm1
	addq	$1, %rax
	movl	%ecx, (%r9,%rdx)
	minss	%xmm0, %xmm1
	cmpq	$1024, %rax
	movss	%xmm1, (%rsi,%rdx)
	jne	L37
	ret
L44:
	movss	LC0(%rip), %xmm0
	jmp	L38
LFE3778:
	.globl _index
	.zerofill __DATA,__pu_bss5,_index,4096,5
	.globl _nn
	.zerofill __DATA,__pu_bss5,_nn,4096,5
	.globl _distNN
	.zerofill __DATA,__pu_bss5,_distNN,4096,5
	.globl _phi
	.zerofill __DATA,__pu_bss5,_phi,4096,5
	.globl _eta
	.zerofill __DATA,__pu_bss5,_eta,4096,5
	.literal4
	.align 2
LC0:
	.long	1203982336
	.literal16
	.align 4
LC1:
	.long	2147483647
	.long	0
	.long	0
	.long	0
	.literal4
	.align 2
LC2:
	.long	1078530011
	.align 2
LC3:
	.long	1086918619
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
	.quad	LFB3773-.
	.set L$set$2,LFE3773-LFB3773
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB3774-.
	.set L$set$4,LFE3774-LFB3774
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB3775-.
	.set L$set$6,LFE3775-LFB3775
	.quad L$set$6
	.byte	0
	.align 3
LEFDE5:
LSFDE7:
	.set L$set$7,LEFDE7-LASFDE7
	.long L$set$7
LASFDE7:
	.long	LASFDE7-EH_frame1
	.quad	LFB3776-.
	.set L$set$8,LFE3776-LFB3776
	.quad L$set$8
	.byte	0
	.align 3
LEFDE7:
LSFDE9:
	.set L$set$9,LEFDE9-LASFDE9
	.long L$set$9
LASFDE9:
	.long	LASFDE9-EH_frame1
	.quad	LFB3778-.
	.set L$set$10,LFE3778-LFB3778
	.quad L$set$10
	.byte	0
	.align 3
LEFDE9:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
