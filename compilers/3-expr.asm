	section	.text
	align	4
	global	main

main:
    push rbp
    mov rbp, rsp
    sub rsp, 24
    mov DWORD [rsp+16], 1
    mov DWORD [rsp+8], 5
    mov DWORD [rsp], 4

    mov rsi, [rsp]
    imul rsi, [rsp+8]
    add rsi, [rsp+16]
    mov rax, rsi

    mov rsp, rbp
    pop rbp
    ret           ; resultado já está em rax
