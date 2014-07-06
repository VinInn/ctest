	.text
	.align 4,0x90
	.globl __Z6scalarRfS_
__Z6scalarRfS_:
LFB221:
	sqrtss	(%rdi), %xmm0
	movss	%xmm0, (%rdi)
	movss	LC0(%rip), %xmm0
	divss	(%rsi), %xmm0
	movss	%xmm0, (%rsi)
	ret
LFE221:
	.align 4,0x90
	.globl __Z4halff
__Z4halff:
LFB222:
	mulss	LC1(%rip), %xmm0
	ret
LFE222:
	.align 4,0x90
	.globl __Z6vectorv
__Z6vectorv:
LFB223:
	movaps	LC2(%rip), %xmm5
	leaq	_v(%rip), %rax
	xorps	%xmm3, %xmm3
	movaps	LC3(%rip), %xmm4
	leaq	_w(%rip), %rdx
	leaq	4096+_v(%rip), %rcx
	.align 4,0x90
L5:
	movaps	(%rax), %xmm1
	movaps	%xmm3, %xmm2
	addq	$16, %rax
	addq	$16, %rdx
	rsqrtps	%xmm1, %xmm0
	cmpneqps	%xmm1, %xmm2
	andps	%xmm2, %xmm0
	mulps	%xmm0, %xmm1
	mulps	%xmm1, %xmm0
	mulps	%xmm4, %xmm1
	addps	%xmm5, %xmm0
	mulps	%xmm1, %xmm0
	movaps	%xmm0, -16(%rax)
	movaps	-16(%rdx), %xmm1
	rcpps	%xmm1, %xmm0
	mulps	%xmm0, %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm0, %xmm0
	subps	%xmm1, %xmm0
	movaps	%xmm0, -16(%rdx)
	cmpq	%rcx, %rax
	jne	L5
	rep; ret
LFE223:
	.globl _w
	.zerofill __DATA,__pu_bss5,_w,4096,5
	.globl _v
	.zerofill __DATA,__pu_bss5,_v,4096,5
	.literal4
	.align 2
LC0:
	.long	1065353216
	.align 2
LC1:
	.long	1050355402
	.literal16
	.align 4
LC2:
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.long	3225419776
	.align 4
LC3:
	.long	3204448256
	.long	3204448256
	.long	3204448256
	.long	3204448256
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
	.quad	LFB221-.
	.set L$set$2,LFE221-LFB221
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
LSFDE3:
	.set L$set$3,LEFDE3-LASFDE3
	.long L$set$3
LASFDE3:
	.long	LASFDE3-EH_frame1
	.quad	LFB222-.
	.set L$set$4,LFE222-LFB222
	.quad L$set$4
	.byte	0
	.align 3
LEFDE3:
LSFDE5:
	.set L$set$5,LEFDE5-LASFDE5
	.long L$set$5
LASFDE5:
	.long	LASFDE5-EH_frame1
	.quad	LFB223-.
	.set L$set$6,LFE223-LFB223
	.quad L$set$6
	.byte	0
	.align 3
LEFDE5:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
