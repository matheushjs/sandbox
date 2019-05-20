	section	.text
	align	4
	global	main

multiply:
	; ao chamar 'call' a CPU empilhou o endereço de retorno
	; então [rsp] contem tal endereço
	mov rdi, QWORD [rsp+8]   ; argumento a
	mov rsi, QWORD [rsp+16]  ; argumento b
	imul rdi, rsi ; realiza multiplicação
	mov rax, rdi  ; resultado vai em rax
	ret
main:
	push rbp      ; salva base pointer
	mov rbp, rsp  ; salva rsp
	sub rsp, 16   ; aloca 16 bytes na stack
	mov QWORD [rsp+8], 5
	mov QWORD [rsp], 4
	call multiply
	mov rsp, rbp  ; restaura rsp
	pop rbp       ; restaura rbp
	ret           ; resultado já está em rax
