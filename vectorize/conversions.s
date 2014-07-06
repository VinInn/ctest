	.section	__TEXT,__text,regular,pure_instructions
	.globl	__Z4ddotDv4_fS_S_
	.align	4, 0x90
__Z4ddotDv4_fS_S_:                      ## @_Z4ddotDv4_fS_S_
	.cfi_startproc
## BB#0:                                ## %entry
	pushq	%rbp
Ltmp2:
	.cfi_def_cfa_offset 16
Ltmp3:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Ltmp4:
	.cfi_def_cfa_register %rbp
	movdqa	%xmm1, %xmm3
	mulss	%xmm2, %xmm3
	pshufd	$1, %xmm2, %xmm4        ## xmm4 = xmm2[1,0,0,0]
	pshufd	$1, %xmm1, %xmm6        ## xmm6 = xmm1[1,0,0,0]
	mulss	%xmm6, %xmm4
	addss	%xmm3, %xmm4
	pshufd	$3, %xmm2, %xmm3        ## xmm3 = xmm2[3,0,0,0]
	movhlps	%xmm2, %xmm2            ## xmm2 = xmm2[1,1]
	pshufd	$3, %xmm1, %xmm8        ## xmm8 = xmm1[3,0,0,0]
	movdqa	%xmm1, %xmm5
	movhlps	%xmm5, %xmm5            ## xmm5 = xmm5[1,1]
	mulss	%xmm5, %xmm2
	pshufd	$1, %xmm0, %xmm7        ## xmm7 = xmm0[1,0,0,0]
	addss	%xmm4, %xmm2
	movdqa	%xmm0, %xmm4
	mulss	%xmm1, %xmm4
	mulss	%xmm6, %xmm7
	mulss	%xmm8, %xmm3
	addss	%xmm2, %xmm3
	addss	%xmm4, %xmm7
	pshufd	$3, %xmm0, %xmm1        ## xmm1 = xmm0[3,0,0,0]
	movhlps	%xmm0, %xmm0            ## xmm0 = xmm0[1,1]
	mulss	%xmm5, %xmm0
	addss	%xmm7, %xmm0
	mulss	%xmm8, %xmm1
	addss	%xmm0, %xmm1
	mulss	%xmm3, %xmm1
	movaps	%xmm1, %xmm0
	popq	%rbp
	ret
	.cfi_endproc

	.globl	__Z4ddotDv4_dS_S_
	.align	4, 0x90
__Z4ddotDv4_dS_S_:                      ## @_Z4ddotDv4_dS_S_
	.cfi_startproc
## BB#0:                                ## %entry
	pushq	%rbp
Ltmp7:
	.cfi_def_cfa_offset 16
Ltmp8:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Ltmp9:
	.cfi_def_cfa_register %rbp
	andq	$-32, %rsp
	subq	$32, %rsp
	movapd	80(%rbp), %xmm4
	movapd	96(%rbp), %xmm1
	movapd	48(%rbp), %xmm2
	movapd	64(%rbp), %xmm8
	movapd	%xmm2, %xmm0
	mulsd	%xmm4, %xmm0
	unpckhpd	%xmm4, %xmm4    ## xmm4 = xmm4[1,1]
	movapd	%xmm2, %xmm3
	unpckhpd	%xmm3, %xmm3    ## xmm3 = xmm3[1,1]
	mulsd	%xmm3, %xmm4
	addsd	%xmm0, %xmm4
	movapd	%xmm8, %xmm6
	mulsd	%xmm1, %xmm6
	movapd	16(%rbp), %xmm7
	movapd	32(%rbp), %xmm0
	movapd	%xmm7, %xmm5
	unpckhpd	%xmm5, %xmm5    ## xmm5 = xmm5[1,1]
	addsd	%xmm4, %xmm6
	mulsd	%xmm2, %xmm7
	mulsd	%xmm3, %xmm5
	unpckhpd	%xmm1, %xmm1    ## xmm1 = xmm1[1,1]
	movapd	%xmm8, %xmm2
	unpckhpd	%xmm2, %xmm2    ## xmm2 = xmm2[1,1]
	mulsd	%xmm2, %xmm1
	addsd	%xmm6, %xmm1
	addsd	%xmm7, %xmm5
	movapd	%xmm0, %xmm3
	mulsd	%xmm8, %xmm3
	addsd	%xmm5, %xmm3
	unpckhpd	%xmm0, %xmm0    ## xmm0 = xmm0[1,1]
	mulsd	%xmm2, %xmm0
	addsd	%xmm3, %xmm0
	mulsd	%xmm1, %xmm0
	movq	%rbp, %rsp
	popq	%rbp
	ret
	.cfi_endproc

	.globl	__Z5ddotdDv4_fS_S_
	.align	4, 0x90
__Z5ddotdDv4_fS_S_:                     ## @_Z5ddotdDv4_fS_S_
	.cfi_startproc
## BB#0:                                ## %entry
	pushq	%rbp
Ltmp12:
	.cfi_def_cfa_offset 16
Ltmp13:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Ltmp14:
	.cfi_def_cfa_register %rbp
	cvtss2sd	%xmm1, %xmm3
	cvtss2sd	%xmm0, %xmm5
	unpcklpd	%xmm3, %xmm5    ## xmm5 = xmm5[0],xmm3[0]
	movapd	%xmm5, %xmm3
	unpckhpd	%xmm3, %xmm3    ## xmm3 = xmm3[1,1]
	pshufd	$1, %xmm2, %xmm6        ## xmm6 = xmm2[1,0,0,0]
	movdqa	%xmm2, %xmm4
	movhlps	%xmm4, %xmm4            ## xmm4 = xmm4[1,1]
	cvtss2sd	%xmm6, %xmm7
	pshufd	$1, %xmm1, %xmm6        ## xmm6 = xmm1[1,0,0,0]
	cvtss2sd	%xmm6, %xmm8
	mulsd	%xmm8, %xmm7
	cvtss2sd	%xmm4, %xmm9
	movdqa	%xmm1, %xmm4
	movhlps	%xmm4, %xmm4            ## xmm4 = xmm4[1,1]
	xorps	%xmm6, %xmm6
	cvtss2sd	%xmm4, %xmm6
	movaps	%xmm0, %xmm4
	movhlps	%xmm4, %xmm4            ## xmm4 = xmm4[1,1]
	cvtss2sd	%xmm4, %xmm4
	mulsd	%xmm6, %xmm4
	mulsd	%xmm6, %xmm9
	xorps	%xmm6, %xmm6
	cvtss2sd	%xmm2, %xmm6
	pshufd	$3, %xmm2, %xmm2        ## xmm2 = xmm2[3,0,0,0]
	unpcklpd	%xmm6, %xmm3    ## xmm3 = xmm3[0],xmm6[0]
	pshufd	$1, %xmm0, %xmm6        ## xmm6 = xmm0[1,0,0,0]
	cvtss2sd	%xmm6, %xmm6
	mulsd	%xmm8, %xmm6
	unpcklpd	%xmm7, %xmm6    ## xmm6 = xmm6[0],xmm7[0]
	mulpd	%xmm5, %xmm3
	cvtss2sd	%xmm2, %xmm2
	pshufd	$3, %xmm1, %xmm1        ## xmm1 = xmm1[3,0,0,0]
	cvtss2sd	%xmm1, %xmm5
	pxor	%xmm1, %xmm1
	addpd	%xmm3, %xmm1
	addpd	%xmm6, %xmm1
	unpcklpd	%xmm9, %xmm4    ## xmm4 = xmm4[0],xmm9[0]
	mulsd	%xmm5, %xmm2
	pshufd	$3, %xmm0, %xmm0        ## xmm0 = xmm0[3,0,0,0]
	cvtss2sd	%xmm0, %xmm0
	mulsd	%xmm5, %xmm0
	addpd	%xmm1, %xmm4
	unpcklpd	%xmm2, %xmm0    ## xmm0 = xmm0[0],xmm2[0]
	addpd	%xmm4, %xmm0
	movapd	%xmm0, %xmm1
	unpckhpd	%xmm1, %xmm1    ## xmm1 = xmm1[1,1]
	mulsd	%xmm0, %xmm1
	xorps	%xmm0, %xmm0
	cvtsd2ss	%xmm1, %xmm0
	popq	%rbp
	ret
	.cfi_endproc

	.globl	__Z5dotd1Dv4_fS_
	.align	4, 0x90
__Z5dotd1Dv4_fS_:                       ## @_Z5dotd1Dv4_fS_
	.cfi_startproc
## BB#0:                                ## %entry
	pushq	%rbp
Ltmp17:
	.cfi_def_cfa_offset 16
Ltmp18:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Ltmp19:
	.cfi_def_cfa_register %rbp
	cvtss2sd	%xmm1, %xmm2
	cvtss2sd	%xmm0, %xmm3
	mulsd	%xmm2, %xmm3
	pshufd	$1, %xmm1, %xmm2        ## xmm2 = xmm1[1,0,0,0]
	cvtss2sd	%xmm2, %xmm4
	pshufd	$1, %xmm0, %xmm2        ## xmm2 = xmm0[1,0,0,0]
	cvtss2sd	%xmm2, %xmm2
	mulsd	%xmm4, %xmm2
	addsd	%xmm3, %xmm2
	pshufd	$3, %xmm1, %xmm3        ## xmm3 = xmm1[3,0,0,0]
	movhlps	%xmm1, %xmm1            ## xmm1 = xmm1[1,1]
	xorps	%xmm4, %xmm4
	cvtss2sd	%xmm1, %xmm4
	pshufd	$3, %xmm0, %xmm1        ## xmm1 = xmm0[3,0,0,0]
	movhlps	%xmm0, %xmm0            ## xmm0 = xmm0[1,1]
	cvtss2sd	%xmm0, %xmm0
	mulsd	%xmm4, %xmm0
	addsd	%xmm2, %xmm0
	xorps	%xmm2, %xmm2
	cvtss2sd	%xmm3, %xmm2
	cvtss2sd	%xmm1, %xmm1
	mulsd	%xmm2, %xmm1
	addsd	%xmm0, %xmm1
	xorps	%xmm0, %xmm0
	cvtsd2ss	%xmm1, %xmm0
	popq	%rbp
	ret
	.cfi_endproc

	.globl	__Z5dotd2Dv4_fS_
	.align	4, 0x90
__Z5dotd2Dv4_fS_:                       ## @_Z5dotd2Dv4_fS_
	.cfi_startproc
## BB#0:                                ## %entry
	pushq	%rbp
Ltmp22:
	.cfi_def_cfa_offset 16
Ltmp23:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Ltmp24:
	.cfi_def_cfa_register %rbp
	cvtss2sd	%xmm1, %xmm2
	cvtss2sd	%xmm0, %xmm3
	mulsd	%xmm2, %xmm3
	pshufd	$1, %xmm1, %xmm2        ## xmm2 = xmm1[1,0,0,0]
	cvtss2sd	%xmm2, %xmm4
	pshufd	$1, %xmm0, %xmm2        ## xmm2 = xmm0[1,0,0,0]
	cvtss2sd	%xmm2, %xmm2
	mulsd	%xmm4, %xmm2
	addsd	%xmm3, %xmm2
	pshufd	$3, %xmm1, %xmm3        ## xmm3 = xmm1[3,0,0,0]
	movhlps	%xmm1, %xmm1            ## xmm1 = xmm1[1,1]
	xorps	%xmm4, %xmm4
	cvtss2sd	%xmm1, %xmm4
	pshufd	$3, %xmm0, %xmm1        ## xmm1 = xmm0[3,0,0,0]
	movhlps	%xmm0, %xmm0            ## xmm0 = xmm0[1,1]
	cvtss2sd	%xmm0, %xmm0
	mulsd	%xmm4, %xmm0
	addsd	%xmm2, %xmm0
	xorps	%xmm2, %xmm2
	cvtss2sd	%xmm3, %xmm2
	cvtss2sd	%xmm1, %xmm1
	mulsd	%xmm2, %xmm1
	addsd	%xmm0, %xmm1
	xorps	%xmm0, %xmm0
	cvtsd2ss	%xmm1, %xmm0
	popq	%rbp
	ret
	.cfi_endproc

	.globl	__Z6dotd21Dv4_fS_
	.align	4, 0x90
__Z6dotd21Dv4_fS_:                      ## @_Z6dotd21Dv4_fS_
	.cfi_startproc
## BB#0:                                ## %entry
	pushq	%rbp
Ltmp27:
	.cfi_def_cfa_offset 16
Ltmp28:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Ltmp29:
	.cfi_def_cfa_register %rbp
	cvtss2sd	%xmm1, %xmm2
	cvtss2sd	%xmm0, %xmm3
	mulsd	%xmm2, %xmm3
	pshufd	$1, %xmm1, %xmm2        ## xmm2 = xmm1[1,0,0,0]
	cvtss2sd	%xmm2, %xmm4
	pshufd	$1, %xmm0, %xmm2        ## xmm2 = xmm0[1,0,0,0]
	cvtss2sd	%xmm2, %xmm2
	mulsd	%xmm4, %xmm2
	addsd	%xmm3, %xmm2
	pshufd	$3, %xmm1, %xmm3        ## xmm3 = xmm1[3,0,0,0]
	movhlps	%xmm1, %xmm1            ## xmm1 = xmm1[1,1]
	xorps	%xmm4, %xmm4
	cvtss2sd	%xmm1, %xmm4
	pshufd	$3, %xmm0, %xmm1        ## xmm1 = xmm0[3,0,0,0]
	movhlps	%xmm0, %xmm0            ## xmm0 = xmm0[1,1]
	cvtss2sd	%xmm0, %xmm0
	mulsd	%xmm4, %xmm0
	addsd	%xmm2, %xmm0
	xorps	%xmm2, %xmm2
	cvtss2sd	%xmm3, %xmm2
	cvtss2sd	%xmm1, %xmm1
	mulsd	%xmm2, %xmm1
	addsd	%xmm0, %xmm1
	xorps	%xmm0, %xmm0
	cvtsd2ss	%xmm1, %xmm0
	popq	%rbp
	ret
	.cfi_endproc

	.globl	__Z5dotd3Dv4_fS_
	.align	4, 0x90
__Z5dotd3Dv4_fS_:                       ## @_Z5dotd3Dv4_fS_
	.cfi_startproc
## BB#0:                                ## %entry
	pushq	%rbp
Ltmp32:
	.cfi_def_cfa_offset 16
Ltmp33:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Ltmp34:
	.cfi_def_cfa_register %rbp
	cvtss2sd	%xmm1, %xmm2
	cvtss2sd	%xmm0, %xmm3
	mulsd	%xmm2, %xmm3
	pshufd	$1, %xmm1, %xmm2        ## xmm2 = xmm1[1,0,0,0]
	cvtss2sd	%xmm2, %xmm4
	pshufd	$1, %xmm0, %xmm2        ## xmm2 = xmm0[1,0,0,0]
	cvtss2sd	%xmm2, %xmm2
	mulsd	%xmm4, %xmm2
	addsd	%xmm3, %xmm2
	pshufd	$3, %xmm1, %xmm3        ## xmm3 = xmm1[3,0,0,0]
	movhlps	%xmm1, %xmm1            ## xmm1 = xmm1[1,1]
	xorps	%xmm4, %xmm4
	cvtss2sd	%xmm1, %xmm4
	pshufd	$3, %xmm0, %xmm1        ## xmm1 = xmm0[3,0,0,0]
	movhlps	%xmm0, %xmm0            ## xmm0 = xmm0[1,1]
	cvtss2sd	%xmm0, %xmm0
	mulsd	%xmm4, %xmm0
	addsd	%xmm2, %xmm0
	xorps	%xmm2, %xmm2
	cvtss2sd	%xmm3, %xmm2
	cvtss2sd	%xmm1, %xmm1
	mulsd	%xmm2, %xmm1
	addsd	%xmm0, %xmm1
	xorps	%xmm0, %xmm0
	cvtsd2ss	%xmm1, %xmm0
	popq	%rbp
	ret
	.cfi_endproc

	.globl	__Z5dotd4Dv4_fS_
	.align	4, 0x90
__Z5dotd4Dv4_fS_:                       ## @_Z5dotd4Dv4_fS_
	.cfi_startproc
## BB#0:                                ## %entry
	pushq	%rbp
Ltmp37:
	.cfi_def_cfa_offset 16
Ltmp38:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
Ltmp39:
	.cfi_def_cfa_register %rbp
	cvtss2sd	%xmm1, %xmm2
	cvtss2sd	%xmm0, %xmm3
	mulsd	%xmm2, %xmm3
	pshufd	$1, %xmm1, %xmm2        ## xmm2 = xmm1[1,0,0,0]
	cvtss2sd	%xmm2, %xmm4
	pshufd	$1, %xmm0, %xmm2        ## xmm2 = xmm0[1,0,0,0]
	cvtss2sd	%xmm2, %xmm2
	mulsd	%xmm4, %xmm2
	addsd	%xmm3, %xmm2
	pshufd	$3, %xmm1, %xmm3        ## xmm3 = xmm1[3,0,0,0]
	movhlps	%xmm1, %xmm1            ## xmm1 = xmm1[1,1]
	xorps	%xmm4, %xmm4
	cvtss2sd	%xmm1, %xmm4
	pshufd	$3, %xmm0, %xmm1        ## xmm1 = xmm0[3,0,0,0]
	movhlps	%xmm0, %xmm0            ## xmm0 = xmm0[1,1]
	cvtss2sd	%xmm0, %xmm0
	mulsd	%xmm4, %xmm0
	addsd	%xmm2, %xmm0
	xorps	%xmm2, %xmm2
	cvtss2sd	%xmm3, %xmm2
	cvtss2sd	%xmm1, %xmm1
	mulsd	%xmm2, %xmm1
	addsd	%xmm0, %xmm1
	xorps	%xmm0, %xmm0
	cvtsd2ss	%xmm1, %xmm0
	popq	%rbp
	ret
	.cfi_endproc


.subsections_via_symbols
