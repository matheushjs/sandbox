    section  .text
    align    4
    global   main
    extern   putchar
normal_func:
    push rbp      ; salva base pointer
    mov rbp, rsp  ; salva stack pointer

    push rsi      ; empilha o 'ep' da lambda (poderia passar pelo rdi)
    call rdi      ; chama a lambda

    mov rsp, rbp  ; restaura stack pointer
    pop rbp       ; restaura base pointer
    ret
lambda:
    push rbp      ; salva base pointer
    mov rbp, rsp  ; salva stack pointer
    sub rsp, 8    ; garante alinhamento 16-bit

    ; [rbp] contem o rbp empilhado, [rbp+8] contem o endereço de retorno
    mov rcx, QWORD [rbp+16]  ; 'ep' passado pela pilha
    mov rdi, QWORD [rcx-8]  ; variável n
    add rdi, 48             ; para imprimir um digito
    call putchar

    mov rsp, rbp  ; restaura stack pointer
    pop rbp       ; restaura base pointer
    ret
main:
    push rbp        ; salva base pointer
    mov rbp, rsp    ; salva stack pointer
    sub rsp, 24     ; aloca espaço na stack
    mov QWORD [rbp-8], 5        ; define variavel n
    mov QWORD [rbp-16], lambda  ; ip da função lambda
    mov QWORD [rbp-24], rbp     ; ep da função lambda

    mov rdi, QWORD [rbp-16]     ; passa o ip da lambda como argumento de normal_func
    mov rsi, QWORD [rbp-24]     ; passa o ep da lambda
    call normal_func

    mov rsp, rbp     ; restaura stack pointer
    pop rbp          ; restaura base pointer
    ret
